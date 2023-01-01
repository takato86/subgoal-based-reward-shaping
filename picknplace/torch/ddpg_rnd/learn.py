import logging
import os
from statistics import mean
from gym.wrappers.record_video import RecordVideo
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from picknplace.torch.ddpg_rnd.replay_memory import DynamicShapingReplayBuffer
from picknplace.utils.constants import device
from picknplace.torch.ddpg.agent import DDPGAgent
from picknplace.torch.ddpg_rnd.agent import RNDAgent
from picknplace.torch.normalizer import Zscorer
from picknplace.utils.mpi import mpi_avg, num_procs, proc_id

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test(test_env, agent, normalizer, pipeline, num_test_episodes, max_ep_len):
    ep_rets, is_successes, n_subgs = [], [], []

    for _ in range(num_test_episodes):
        f_o, d, rets, ep_len = test_env.reset(), False, [], 0
        o, _, _, _, _, _ = pipeline.transform(f_o, None, 0, None, False, None)

        while(not d and (ep_len < max_ep_len)):
            n_o = normalizer(o)
            a = agent.act(n_o, noise_scale=0, epsilon=0)
            f_o2, r, d, info = test_env.step(a)
            o, a, r, o2, d, info = pipeline.transform(f_o, a, r, f_o2, d, info)
            f_o, o = f_o2.copy(), o2.copy()
            rets.append(r)
            ep_len += 1

        ep_rets.append(sum(rets))
        is_successes.append(bool(info["is_success"]))
        n_subgs.append(info["subgoal"])

    # logger.info(f"test return: {ep_ret}")
    test_env.reset()
    return {
        "Test/return": mean(ep_rets),
        "Test/succ_rate": mean(is_successes),
        "Test/n_subgs": mean(n_subgs)
    }


def learn(env_fn, pipeline, test_pipeline, replay_buffer_fn, epochs, steps_per_epoch, start_steps, update_after,
          update_every, num_test_episodes, max_ep_len, gamma, epsilon, actor_lr, critic_lr, polyak, l2_action,
          noise_scale, batch_size, norm_clip, norm_eps, clip_return, is_pos_return, feature_size, rnd_lr, logdir=None,
          video=False):
    env = env_fn()
    test_env = env

    if proc_id() == 0:
        video_folder = os.path.join(logdir, "videos")
        test_env = RecordVideo(env, video_folder=video_folder) if video else env
        writer = SummaryWriter(logdir)

    total_steps = steps_per_epoch * epochs // num_procs()
    f_o, ep_ret, ep_len, total_test_ep_ret, num_episodes, is_succ = env.reset(), 0, 0, 0, 0, False
    o, _, _, _, _, _ = pipeline.transform(f_o, None, 0, None, False, None)
    agent = DDPGAgent(
        o, env.action_space, gamma, actor_lr, critic_lr,
        polyak, l2_action, clip_return, is_pos_return, logger
    )
    rnd_agent = RNDAgent(o, feature_size, rnd_lr)  # move feature_size and rnd_lr into config
    normalizer = Zscorer(norm_clip, norm_eps)
    replay_buffer = replay_buffer_fn(
        o.shape[1], env.action_space.shape[0]
    )

    # 入出力の型をモジュール毎に統一したい。
    for i in tqdm(range(int(total_steps))):

        if i > start_steps:
            n_o = normalizer(o)
            a = agent.act(n_o, noise_scale, epsilon)
        else:
            a = env.action_space.sample()

        f_o2, r, d, info = env.step(a)
        o, a, r, o2, d, info = pipeline.transform(f_o, a, r, f_o2, d, info)
        bonus = rnd_agent.act(o)
        replay_buffer.store(o, a, r, o2, d, bonus, info)
        f_o, o = f_o2.copy(), o2.copy()
        ep_len, ep_ret = ep_len + 1, ep_ret + r

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.info(ep_len, ep_ret)
            is_succ = bool(info["is_success"])
            num_episodes += 1
            n_subgs = info["subgoal"]
            avg_ep_ret, avg_is_succ = mpi_avg(ep_ret), mpi_avg(is_succ)
            avg_subgs = mpi_avg(n_subgs)

            if proc_id() == 0:
                writer.add_scalar("Train/return", scalar_value=avg_ep_ret, global_step=num_episodes)
                writer.add_scalar("Train/succ_rate", scalar_value=avg_is_succ, global_step=num_episodes)
                writer.add_scalar("Train/n_subgs", scalar_value=avg_subgs, global_step=num_episodes)

            f_o, ep_ret, ep_len, is_succ = env.reset(), 0, 0, False
            o, _, r, _,  d, _ = pipeline.transform(f_o, None, 0, None, False, None)

        # Update handling
        if i >= update_after and i % update_every == 0:
            logger.debug("Start to learn from experiences.")
            basis = (i - update_after) // update_every
            for j in range(update_every):
                logger.debug("Getting experiences...")
                batch = replay_buffer.sample_batch(batch_size=batch_size)
                logger.debug("Got experiences!")
                bonus = batch.pop("bonus")

                # update rnd
                rnd_batch = batch["obs"]
                logger.debug("Updating RND...")
                rnd_loss = rnd_agent.update(rnd_batch)
                logger.debug("Updated RND!")
                
                batch["rew"] += bonus
                logger.debug("Normalize observations...")
                batch["obs"] = normalizer(batch["obs"].cpu()).to(device())
                batch["obs2"] = normalizer(batch["obs2"].cpu()).to(device())
                logger.debug("Normalized observations!")
                logger.debug("Updating...")
                loss_q, loss_pi, max_q = agent.update(batch)
                logger.debug("Updated!")

                if (basis + j) % steps_per_epoch == 0:
                    n_records = (basis + j) // steps_per_epoch
                    avg_loss_q, avg_loss_pi, avg_max_q = mpi_avg(loss_q), mpi_avg(loss_pi), mpi_avg(max_q)
                    avg_rnd_loss = mpi_avg(rnd_loss)
                    if proc_id() == 0:
                        writer.add_scalar("Train/q_avg", avg_max_q, n_records)
                        writer.add_scalar("Train/loss_q", avg_loss_q, n_records)
                        writer.add_scalar("Train/loss_pi", avg_loss_pi, n_records)
                        writer.add_scalar("Train/loss_rnd", avg_rnd_loss, n_records)

                        if type(replay_buffer) == DynamicShapingReplayBuffer and hasattr(replay_buffer.shaper, "vfunc"):
                            values = {str(key): value for key, value in replay_buffer.shaper.vfunc.state_dict().items()}
                            writer.add_scalars("Train/potentials", values, n_records)

            agent.sync_target()
            logger.debug("Finish to learn from experiences.")

        # End of epoch handling
        if (i+1) % steps_per_epoch == 0:
            epoch = (i+1) // steps_per_epoch
            logger.debug(f"Finish epoch: {epoch}.")
            # logger.info(f"Epoch {epoch}\n-------------------------------")
            # logger.info(f"return: {ep_ret}   [{i:>7d}/{int(total_steps):>7d}]")
            score_dict = test(
                test_env, agent, normalizer, test_pipeline, num_test_episodes, max_ep_len
            )

            for key, score in score_dict.items():
                avg_score = mpi_avg(score)

                if proc_id() == 0:
                    writer.add_scalar(key, avg_score, epoch)

                if key == "Test/return":
                    total_test_ep_ret += score

    if proc_id() == 0:
        agent.save(os.path.join(writer.log_dir, "model.pth"))
        writer.close()

    return total_test_ep_ret

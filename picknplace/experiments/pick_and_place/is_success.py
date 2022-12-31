

def is_success(done, info):
    # return done
    # if not done:
    #     return False
    return bool(info.get("is_success"))


def is_success4sarsa_rs(done, info):
    # return done
    # if not done:
    # return done and bool(info.get("is_success"))
    return done
    # return bool(info.get("is_success"))


def is_success4dta(done, info):
    return bool(info.get("is_success"))

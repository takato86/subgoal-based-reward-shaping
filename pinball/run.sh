for JSON in "srs.json" "static_random.json"
do
echo "in/configs/${JSON}"
xvfb-run -a python main.py --config="in/configs/${JSON}"
done

# Create problem data using rudy
#
# To run this script use the following command from the top of the project
#
# ```
# bash data/rudy/create_problem_data.sh
# ```

mkdir -p out

for density in 5 10 15 20 ; do
    for size in 1000 2000 3000 4000 5000 ; do
        for seed in 1 2 3 4; do
            python data/rudy/interface.py $size $density $seed 1 40 > data/rudy/out/weighted_graph_${size}_${density}_${seed}.dat-s
        done
    done
done

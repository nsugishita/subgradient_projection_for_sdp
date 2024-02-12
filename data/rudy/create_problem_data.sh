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
            python data/rudy/create_max_cut.py $size $density $seed > data/rudy/out/graph_${size}_${density}_${seed}.dat-s
        done
    done
done

for density in 5 10 15 20 ; do
    for size in 1000 2000 3000 4000 5000 ; do
        for seed in 1 2 3 4; do
            python data/rudy/create_gpp.py $size $density $seed > data/rudy/out/gpp_${size}_${density}_${seed}.dat-s
        done
    done
done

# Run SDPNAL+
#
# One can specify the problem name and the number of iterations. Optionally
# the final solution can be saved in a file. To use this script, run the
# following command from the top of the project directory (where 'external'
# directory resides).
#
# ```
# bash scripts/sdpnal/interface.sh PROBLEMNAME ITERATION [RESULTFILEPATH]
# ```
#
# For example, the following command run SDPNAL+ on `gpp100` for 200 iterations,
# and save the solution in `outs/solution.txt`.
#
# ```
# bash scripts/sdpnal/interface.sh gpp100 200 outs/solution.txt
# ```

export OMP_NUM_THREADS=1

problem=$1
iteration=$2
solution_path=$3

cd external/SDPNAL+v1.0/
module load matlab
if [ "$#" -eq 3 ]; then
    matlab -nodisplay -nosplash -nodesktop -r "maxNumCompThreads(1); startup; [blk,At,C,b] = read_sdpa('../../data/SDPLIB/data/${problem}.dat-s'); OPTIONS.maxiter=$iteration; OPTIONS.tol=0; OPTIONS.stopoption=0; [obj,X,y,Z] = sdpnalplus(blk,At,C,b,[],[],[],[],[],OPTIONS); save('${solution_path}', 'Z', '-ascii');  exit;"
else
    matlab -nodisplay -nosplash -nodesktop -r "maxNumCompThreads(1); startup; [blk,At,C,b] = read_sdpa('../../data/SDPLIB/data/${problem}.dat-s'); OPTIONS.maxiter=$iteration; OPTIONS.tol=0; OPTIONS.stopoption=0; [obj,X,y,Z] = sdpnalplus(blk,At,C,b,[],[],[],[],[],OPTIONS); exit;"
fi

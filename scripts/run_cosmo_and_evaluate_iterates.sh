if [ "$1" == "1" ]; then
    problems="gpp250-1 gpp500-1 gpp250-2 gpp500-2"
fi
if [ "$1" == "2" ]; then
    problems="gpp250-3 gpp500-3 gpp250-4 gpp500-4"
fi
if [ "$1" == "3" ]; then
    problems="mcp250-1 mcp500-1 mcp250-2 mcp500-2"
fi
if [ "$1" == "4" ]; then
    problems="mcp250-3 mcp500-3 mcp250-4 mcp500-4"
fi
if [ "$1" == "5" ]; then
    problems="mcp100"
fi

module load julia
export OMP_NUM_THREADS=1
julia --project=juliaenv examples/run_cosmo_and_evaluate_iteration.jl $problems

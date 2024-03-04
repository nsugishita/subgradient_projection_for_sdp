using SparseArrays

# Read a file of the SDPA file format into a dict of the following items.
# - m: dimension of x
# - n: dimension of Fs
# - n_blocks: number of blocks in the diagonal structure of the matrices
# - block_sizes: vector of numbers that give the sizes on the individual blocks
# - c: objective vector
# - F: array containing the constraint matrices. F[i,j] is the jth block of
#      Fi. Note that F[1,1] corresponds to the first block of F0 in the SDPA
#      file format.
function read_sdpa_file(input_file)
  if !isfile(input_file)
      if isfile(splitext(input_file)[1] * ".dat-s")
          input_file = splitext(input_file)[1] * ".dat-s";
      end
  end
  # open file and read everything into memory
  f = open(input_file);
  lines = readlines(f);
  close(f);

  # initialize variables
  counter = 1;
  # size of x
  m =  NaN;
  # size of symmetric F matrices
  n = NaN;
  n_blocks = NaN;
  block_sizes = zeros(1);
  c = zeros(0);
  F = Array{SparseMatrixCSC}(undef, 0, 0);

  matrix_number_list = Vector{Int}();
  block_number_list = Vector{Int}();
  i_list = Vector{Int}();
  j_list = Vector{Int}();
  entry_list = Vector{Float64}();

  for ln in lines
    if length(ln) == 0
      continue
    end
    # dont do anything if line contains a comment
    if ln[1] == '"' || ln[1] == "*"
      counter == 1
    else
      if counter == 1
        m = parse(Int64,strip(ln));

      elseif counter == 2
        # n_blocks: number of blocks in the diagonal structure of the matrices
        n_blocks = parse(Int64,strip(ln))

      elseif counter == 3
        # vector of numbers that give the sizes on the individual blocks
        # negative number indicates diagonal submatrix
        bvec = split(replace(strip(ln,['{','}','(',')']),"+" =>""))
        block_sizes = [parse(Float64, ss) for ss in bvec]
        n = Int(sum(abs.(block_sizes)))

      elseif counter == 4
        # objective function vector c
        # try first to split by whitespace as delimiter (also remove certain trailing, ending characters)
        cvec = split(replace(strip(ln,['{','}','(',')']),"+" => ""))
        # otherwise try comma as delimiter
        if length(cvec) == 1
          cvec = split(replace(strip(ln,['{','}','(',')']),"+" => ""),",")
        end
        c = [parse(Float64, ss) for ss in cvec]

      else
        # all other lines contain constraint matrices with one entry per line
        # save them directly as sparse matrix
        line = split(ln)
        # We need to adjust matrix_number since it is 0-based.
        matrix_number = parse(Int, line[1]) + 1;
        block_number = parse(Int, line[2]);
        i = parse(Int, line[3]);
        j = parse(Int, line[4]);
        entry = parse(Float64, line[5]);
        if i == j
            push!(matrix_number_list, matrix_number);
            push!(block_number_list, block_number);
            push!(i_list, i);
            push!(j_list, j);
            push!(entry_list, entry);
        else
            push!(matrix_number_list, matrix_number);
            push!(block_number_list, block_number);
            push!(i_list, i);
            push!(j_list, j);
            push!(entry_list, entry);
            push!(matrix_number_list, matrix_number);
            push!(block_number_list, block_number);
            push!(i_list, j);
            push!(j_list, i);
            push!(entry_list, entry);
        end
      end
      counter += 1
    end
  end

  F = Array{SparseMatrixCSC}(undef, n_blocks, m + 1);

  for block_index in 1:n_blocks
      for matrix_index in 1:(m + 1)
          selector = (
             (matrix_number_list .== matrix_index)
             .& (block_number_list .== block_index));
         F[block_index, matrix_index] = sparse(
           i_list[selector], j_list[selector], entry_list[selector],
           block_sizes[block_index], block_sizes[block_index]
          );
      end
  end

  return Dict("m" => m, "n" => n, "n_blocks" => n_blocks, "block_sizes" => block_sizes, "c" => c, "F" => F);
end

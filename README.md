# How to use

- Copy the test file to `DASHFORMER_D10/dashformer/data/example_AA_sequences.list`
- Open a terminal and enter the `DASHFORMER_D10` directory
- Run 
  
        docker buildx build -t idash .
- Run
        
        docker run -it idash bash
- Run
    
        go build
- Run 
    
        ./dashformer

# Get results

- The results are now in the file `/home/data/output/output.txt`
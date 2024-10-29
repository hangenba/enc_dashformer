# How to use

- Copy the test file to `enc_dashformer/dashformer/data/example_AA_sequences.list`
- Open a terminal and enter the `enc_dashformer` directory
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

# Get data

- The address of data: [IDASH24](https://drive.google.com/drive/folders/13_a4H3pkwi36lJOqh4rgW0odKcVXrQ2S)

- File Directory Tree:
  - `data`
    - `dashformer_model_parameters` 
      - ...parameters
    - `output`
    - `dashformer_tokenizer.json`
    - `example_AA_sequences.list`

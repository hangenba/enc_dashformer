package utils

/*
* This package include read file function
* 1. ReadWordIndex: reads a JSON file from the given path
* 2. ReadExampleData: read example file from the give path
 */

import (
	"bufio"
	"dashformer/config"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

// 读取 JSON 文件并解析 word_index 字段的函数
func ReadWordIndex(filePath string) (map[string]int, error) {
	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// 读取文件内容
	byteValue, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	// 定义一个空接口来存储解码后的 JSON 数据
	var result map[string]interface{}
	if err := json.Unmarshal(byteValue, &result); err != nil {
		return nil, err
	}

	// 提取 config 字段
	config, ok := result["config"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("config field not found or is not a JSON object")
	}

	// 提取 word_index 字段
	wordIndexStr, ok := config["word_index"].(string)
	if !ok {
		return nil, fmt.Errorf("word_index field not found or is not a string")
	}

	// 将 word_index 字段解析为 map[string]int
	var wordIndex map[string]int
	if err := json.Unmarshal([]byte(wordIndexStr), &wordIndex); err != nil {
		return nil, err
	}

	return wordIndex, nil
}

// ReadExampleData 读取文件并将其转换为one-hot编码的三维张量
func ReadExampleData(filePath string, tokenizer map[string]int) ([][][]float64, error) {
	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("无法打开文件: %v", err)
	}
	defer file.Close()

	// 存储字符到数字的映射
	numMatrix := [][][]float64{}

	// 扫描文件的每一行
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ",")
		chars := strings.Split(parts[0], " ")
		nums := []int{}

		for _, char := range chars {
			char = strings.ToLower(char) // 将字符转换为小写字母
			if num, exists := tokenizer[char]; exists {
				nums = append(nums, num)
			} else {
				log.Fatalf("词向量映射错误: %v", err)
				return [][][]float64{}, nil
			}
		}

		// 将每一行的数字列表转换为one-hot编码
		oneHotMatrix := make([][]float64, len(nums))
		for i, num := range nums {
			oneHotVec := make([]float64, len(tokenizer)+1) // 生成one-hot向量
			oneHotVec[num] = 1.0
			oneHotMatrix[i] = oneHotVec
		}

		numMatrix = append(numMatrix, oneHotMatrix)
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("读取文件时出错: %v", err)
	}
	// // 输出三维张量
	// fmt.Println("三维张量:")
	// for _, matrix := range numMatrix {
	// 	for _, row := range matrix {
	// 		fmt.Println(row)
	// 	}
	// }
	return numMatrix, nil
}

// 将文件转换成二维切片
func parseMatrix(lines []string) ([][]float64, error) {
	var matrixSlices [][]float64

	for i, line := range lines {
		cols := strings.Fields(line)
		row := make([]float64, len(cols))
		for j, col := range cols {
			num, err := strconv.ParseFloat(col, 64)
			if err != nil {
				return [][]float64{}, fmt.Errorf("error converting string to float64 at line %d, column %d: %v", i+1, j+1, err)
			}
			row[j] = num
		}
		matrixSlices = append(matrixSlices, row)
	}

	return matrixSlices, nil
}

// 将文件转换成一维向量
func parseVector(lines []string) ([]float64, error) {
	var vectorSlices []float64

	for i, line := range lines {
		cols := strings.Fields(line)
		if len(cols) != 1 {
			return []float64{}, fmt.Errorf("line %d does not contain 1 column", i+129)
		}
		num, err := strconv.ParseFloat(cols[0], 64)
		if err != nil {
			return []float64{}, fmt.Errorf("error converting string to float64: %v", err)
		}
		vectorSlices = append(vectorSlices, num)
	}

	return vectorSlices, nil
}

// 读取W_Q, W_K, W_V 三个文件
func ReadMultiAttentionFile(filename string) ([4][][]float64, [4][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) != 256 {
		return [4][][]float64{}, [4][]float64{}, fmt.Errorf("the file does not contain 256 lines")
	}

	// 解析前128行
	var matrixSlices [4][][]float64
	for i := 0; i < 4; i++ {
		matrixSlices[i] = make([][]float64, 128)
		for j := range matrixSlices[i] {
			matrixSlices[i][j] = make([]float64, 32)
		}
	}

	for i := 0; i < 128; i++ {
		cols := strings.Fields(lines[i])
		if len(cols) != 128 {
			return [4][][]float64{}, [4][]float64{}, fmt.Errorf("line %d does not contain 128 columns", i+1)
		}
		for j := 0; j < 128; j++ {
			num, err := strconv.ParseFloat(cols[j], 64)
			if err != nil {
				return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error converting string to float64: %v", err)
			}
			matrixSlices[j/32][i][j%32] = num
		}
	}

	// 解析后128行
	var MatrixSlices [4][]float64
	for i := 0; i < 4; i++ {
		MatrixSlices[i] = make([]float64, 32)
	}

	for i := 0; i < 128; i++ {
		cols := strings.Fields(lines[i+128])
		if len(cols) != 1 {
			return [4][][]float64{}, [4][]float64{}, fmt.Errorf("line %d does not contain 1 column", i+129)
		}
		num, err := strconv.ParseFloat(cols[0], 64)
		if err != nil {
			return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error converting string to float64: %v", err)
		}
		MatrixSlices[i/32][i%32] = num
	}

	return matrixSlices, MatrixSlices, nil
}

// 读取文件combineHead和classifier
func ReadCombineAndClassifierFile(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [][]float64{}, []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) <= 128 {
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 取前128行
	first128Lines := lines[:128]

	// 取128行后的所有行
	remainingLines := lines[128:]

	// 解析前128行
	matrixWeightSlices, err := parseMatrix(first128Lines)
	if err != nil {
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// fmt.Printf("First 128 lines parsed successfully with dimensions: %d rows x varying columns\n", len(matrixWeightSlices))

	// 解析128行后的所有行
	vectorBaisSlices, err := parseVector(remainingLines)
	if err != nil {
		fmt.Println("Error parsing remaining lines:", err)
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	return matrixWeightSlices, vectorBaisSlices, nil
}

// 读取文件combineHead和classifier
func ReadFeedFowardFile(filename string) ([][]float64, []float64, [][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) <= 128 {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 取W_1和b_1
	weightMatrixLines1 := lines[:128]
	biasVectorLines1 := lines[128:384]

	// 取W_2和b_2
	weightMatrixLines2 := lines[384:640]
	biasVectorLines2 := lines[640:]

	// 解析W_1和b_1
	matrixWeightSlice1, err := parseMatrix(weightMatrixLines1)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}
	vectorBaisSlice1, err := parseVector(biasVectorLines1)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 解析W_2和b_2
	matrixWeightSlice2, err := parseMatrix(weightMatrixLines2)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}
	vectorBaisSlice2, err := parseVector(biasVectorLines2)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	return matrixWeightSlice1, vectorBaisSlice1, matrixWeightSlice2, vectorBaisSlice2, nil
}

// 读取文件LayerNorm
func ReadLayerNormFile(filename string) ([]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return []float64{}, []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) <= 128 {
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 取前128行
	first128Lines := lines[:128]

	// 取128行后的所有行
	remainingLines := lines[128:]

	// 解析前128行
	vectorRSlices, err := parseVector(first128Lines)
	if err != nil {
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// fmt.Printf("First 128 lines parsed successfully with dimensions: %d rows x varying columns\n", len(matrixWeightSlices))

	// 解析128行后的所有行
	vectorBSlices, err := parseVector(remainingLines)
	if err != nil {
		fmt.Println("Error parsing remaining lines:", err)
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	return vectorRSlices, vectorBSlices, nil
}

// 读取文件LayerNorm
func ReadLayerNormSqrtVarianceFile(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 解析所有行
	vectorVaranceSlices, err := parseVector(lines)
	if err != nil {
		return []float64{}, fmt.Errorf("error reading file: %v", err)
	}
	return vectorVaranceSlices, nil
}

// 读取文件embedding和encoding
func ReadEcodeingFile(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [][]float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	// 使用bufio.Scanner逐行读取文件
	scanner := bufio.NewScanner(file)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	// 检查是否有读取错误
	if err := scanner.Err(); err != nil {
		return [][]float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 将文件读成矩阵
	matrixSlices, err := parseMatrix(lines)
	if err != nil {
		return [][]float64{}, fmt.Errorf("error reading file: %v", err)
	}
	return matrixSlices, nil
}

// 读取Transformer参数文件
func ReadModelParameterFile(fileDir string) (DashformerModelParameters, error) {

	// 读取文件embedding
	embeddingData, err := ReadEcodeingFile(fileDir + "/embedding_Embedding_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("embedding: (%d , %d)\n", len(embeddingData), len(embeddingData[0]))

	// 读取文件encoding
	encodingData, err := ReadEcodeingFile(fileDir + "/positional_encoding_Lookup.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("encoding: (%d , %d)\n", len(encodingData), len(encodingData[0]))

	// 读取文件W_Q, W_K, W_V
	queryWeightMatrixs, queryBiasVectors, err := ReadMultiAttentionFile(fileDir + "/transformer_block_Query_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("queryWeightMatrixs: (%d, %d, %d)   ", len(queryWeightMatrixs), len(queryWeightMatrixs[0]), len(queryWeightMatrixs[0][0]))
	// fmt.Printf("queryBiasVectors: (%d, %d, )\n", len(queryBiasVectors), len(queryBiasVectors[0]))

	keyWeightMatrixs, keyBiasVectors, err := ReadMultiAttentionFile(fileDir + "/transformer_block_Key_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("keyWeightMatrixs: (%d, %d, %d)   ", len(keyWeightMatrixs), len(keyWeightMatrixs[0]), len(keyWeightMatrixs[0][0]))
	// fmt.Printf("keyBiasVectors: (%d, %d, )\n", len(keyBiasVectors), len(keyBiasVectors[0]))

	valueWeightMatrixs, valueBiasVectors, err := ReadMultiAttentionFile(fileDir + "/transformer_block_Value_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("valueWeightMatrixs: (%d, %d, %d)   ", len(valueWeightMatrixs), len(valueWeightMatrixs[0]), len(valueWeightMatrixs[0][0]))
	// fmt.Printf("valueBiasVectors: (%d, %d, )\n", len(valueBiasVectors), len(valueBiasVectors[0]))

	// 读取文件combineHead和classifier
	combineWeightMatrixs, combineBiasVectors, err := ReadCombineAndClassifierFile(fileDir + "/transformer_block_CombineHead_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("combineWeightMatrixs: (%d, %d)   ", len(combineWeightMatrixs), len(combineWeightMatrixs[0]))
	// fmt.Printf("combineBiasVectors: (%d, )\n", len(combineBiasVectors))

	classifierWeightMatrixs, classifierBiasVectors, err := ReadCombineAndClassifierFile(fileDir + "/Dense_Classifier_DenseClassifier_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("classifierWeightMatrixs: (%d, %d)   ", len(classifierWeightMatrixs), len(classifierWeightMatrixs[0]))
	// fmt.Printf("classifierBiasVectors: (%d, )\n", len(classifierBiasVectors))

	// 读取文件LayerNorm
	LayerNormMatrixsR1, LayerNormMatrixsB1, err := ReadLayerNormFile(fileDir + "/transformer_block_LayerNorm1_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("LayerNormMatrixsR1: (%d, )   ", len(LayerNormMatrixsR1))
	// fmt.Printf("LayerNormMatrixsB1: (%d, )\n", len(LayerNormMatrixsB1))

	LayerNormMatrixsR2, LayerNormMatrixsB2, err := ReadLayerNormFile(fileDir + "/transformer_block_LayerNorm2_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	// fmt.Printf("LayerNormMatrixsR1: (%d, )   ", len(LayerNormMatrixsR2))
	// fmt.Printf("LayerNormMatrixsB1: (%d, )\n", len(LayerNormMatrixsB2))

	// 读取文件FeedForward
	feedForwardWeightMatrix1, feedForwardBiasVector1, feedForwardWeightMatrix2, feedForwardBiasVector2, err := ReadFeedFowardFile(fileDir + "/transformer_block_FFN_weights.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}

	// dashModelParam.PrintDimensions()

	// 设置多项式拟合系数
	reluCoefficients, sqrtLayerCoefficients1, sqrtLayerCoefficients2, softMaxB, softMaxC := config.InitCoeffients()
	layerNormSqrtVariance1, err := ReadLayerNormSqrtVarianceFile(fileDir + "/layerNorm1_Reciprocal_SqrtVariance.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}
	layerNormSqrtVariance2, err := ReadLayerNormSqrtVarianceFile(fileDir + "/layerNorm2_Reciprocal_SqrtVariance.txt")
	if err != nil {
		log.Fatalf("文件读取失败: %v", err)
	}

	return DashformerModelParameters{
		EmbeddingMatrix: embeddingData,
		EncodingMatrix:  encodingData,

		QueryWeightAttentionMatrixs: queryWeightMatrixs,
		QueryBiasAttentionVectors:   queryBiasVectors,
		KeyWeightAttentionMatrixs:   keyWeightMatrixs,
		KeyBiasAttentionVectors:     keyBiasVectors,
		ValueWeightAttentionMatrixs: valueWeightMatrixs,
		ValueBiasAttentionVectors:   valueBiasVectors,

		CombineWeightMatrixs: combineWeightMatrixs,
		CombineBiasVectors:   combineBiasVectors,

		LayerNormVectorR1: LayerNormMatrixsR1,
		LayerNormVectorB1: LayerNormMatrixsB1,
		LayerNormVectorR2: LayerNormMatrixsR2,
		LayerNormVectorB2: LayerNormMatrixsB2,

		FeedForwardWeightMatrix1: feedForwardWeightMatrix1,
		FeedForwardBiasVector1:   feedForwardBiasVector1,
		FeedForwardWeightMatrix2: feedForwardWeightMatrix2,
		FeedForwardBiasVector2:   feedForwardBiasVector2,

		ClassifierWeightMatrix: classifierWeightMatrixs,
		ClassifierBiasVector:   classifierBiasVectors,

		ReluCoefficients:       reluCoefficients,
		SqrtLayerCoefficients1: sqrtLayerCoefficients1,
		SqrtLayerCoefficients2: sqrtLayerCoefficients2,

		LayerNormSqrtVariance1: layerNormSqrtVariance1,
		LayerNormSqrtVariance2: layerNormSqrtVariance2,
		SoftMaxB:               softMaxB,
		SoftMaxC:               softMaxC,
	}, nil
}

// 写三维张量到文件中
func WriteResultToFile(fileDir string, valueTensor [][][]float64) {
	// 打开文件用于写入
	file, err := os.Create(fileDir + "/output.txt")
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	// 遍历 valueTensor 并写入文件
	for i := range valueTensor {
		for j := range valueTensor[i][0] {
			if j != 0 {
				fmt.Fprint(file, "\t") // 在每个值之间添加制表符作为分隔符
			}
			valueTensor[i][0][j] = valueTensor[i][0][j] * 2649.372705
			fmt.Fprint(file, valueTensor[i][0][j])
		}
		fmt.Fprintln(file) // 每行结束后写入一个换行符
	}
}

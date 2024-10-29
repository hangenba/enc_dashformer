package main

import (
	"dashformer/coefficient"
	"dashformer/config"
	"dashformer/encryption"
	"dashformer/maths"
	"dashformer/utils"
	"fmt"
	"log"
	"time"
)

// func evalDashformerMultiTread(publicKeys *encryption.PublicParametersKeys, secretKeys *encryption.SecretParametersKeys,
// 	exampleData [][][]float64, dashModelParam utils.DashformerModelParameters) (*encryption.CiphertextTensor, error) {
// 	// 2.2.加密example数据
// 	ciphertextTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, exampleData)
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Printf("Encrypte data already, Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)

// 	// 3.1.Embedding
// 	startTime := time.Now()
// 	embeddingCipherTensor, err := maths.CiphertextTensorMultiplyPlaintextMatrixMultiThread(publicKeys, ciphertextTensor, dashModelParam.EmbeddingMatrix)
// 	if err != nil {
// 		panic(err)
// 	}
// 	elapsedTime := time.Since(startTime)
// 	fmt.Printf("Embedding took %s to run.\n", elapsedTime)

// 	// 3.2.Encoding
// 	startTime = time.Now()
// 	encodingCipherTensor, err := maths.CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys, embeddingCipherTensor, dashModelParam.EncodingMatrix)
// 	if err != nil {
// 		panic(err)
// 	}
// 	elapsedTime = time.Since(startTime)
// 	fmt.Printf("Encoding took %s to run.\n", elapsedTime)

// 	var concatenateHeader *encryption.CiphertextTensor
// 	// 3.3.Attention
// 	startTime = time.Now()
// 	for i := 0; i < 4; i++ {
// 		// 3.3.1. Compute Q,K,V
// 		multiAttentionQ, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, encodingCipherTensor, dashModelParam.QueryWeightAttentionMatrixs[i], dashModelParam.QueryBiasAttentionVectors[i])
// 		if err != nil {
// 			panic(err)
// 		}

// 		multiAttentionK, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, encodingCipherTensor, dashModelParam.KeyWeightAttentionMatrixs[i], dashModelParam.KeyBiasAttentionVectors[i])
// 		if err != nil {
// 			panic(err)
// 		}

// 		multiAttentionV, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, encodingCipherTensor, dashModelParam.ValueWeightAttentionMatrixs[i], dashModelParam.ValueBiasAttentionVectors[i])
// 		if err != nil {
// 			panic(err)
// 		}

// 		// // 3.3.2 Compute Q X K^T --> Halevi-Shoup
// 		// // **为了优化矩阵乘法层数，将softMax(X)=(x+0.95)^2/400 --> softMax(X)= (QXK^T/(sqrt(32)*20)+ 0.95/20 )^2 **
// 		// multiAttentionQMulKT, err := maths.CiphertextTensorMultiplyCiphertextTensorToHalveiShoupMultiThread(publicKeys, multiAttentionQ, multiAttentionK, 1/(math.Sqrt(32)*20))
// 		// if err != nil {
// 		// 	panic(err)
// 		// }

// 		// // 3.3.3 Compute SoftMax
// 		// multiAttentionQMulKTSoftMax, err := maths.ApproximateSoftmax(publicKeys, multiAttentionQMulKT, 0.95/20, 1)
// 		// if err != nil {
// 		// 	panic(err)
// 		// }

// 		// // 3.3.4 Compute softMax(Q X K^T/sqrt(32)) X V --> X
// 		// multiAttentionHeader, err := maths.CiphertextTensorHSMultiplyCiphertextTensorMultiThread(publicKeys, multiAttentionQMulKTSoftMax, multiAttentionV)
// 		// if err != nil {
// 		// 	panic(err)
// 		// }

// 		multiAttentionHeader, err := maths.CiphertextTensorQKVToAttentionWithBSGSMultiThread(publicKeys, multiAttentionQ, multiAttentionK, multiAttentionV, 7, 8, dashModelParam.SoftMaxB[i], dashModelParam.SoftMaxC[i])
// 		if err != nil {
// 			panic(err)
// 		}
// 		// 3.3.5 concatenate header
// 		concatenateHeader, err = encryption.MergeAndAddCiphertextTensors(concatenateHeader, multiAttentionHeader)
// 		if err != nil {
// 			panic(err)
// 		}
// 	}
// 	elapsedTime = time.Since(startTime)
// 	fmt.Printf("Attention took %s to run.\n", elapsedTime)

// 	// 3.3. Combining Header
// 	startTime = time.Now()
// 	combiningHeader, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, concatenateHeader, dashModelParam.CombineWeightMatrixs, dashModelParam.CombineBiasVectors)
// 	if err != nil {
// 		panic(err)
// 	}
// 	elapsedTime = time.Since(startTime)
// 	fmt.Printf("Combining Header took %s to run.\n", elapsedTime)

// 	// 3.4. Add and LayNorm-1  ***近似1/x需要在layerNorm函数内部进行修改
// 	startTime = time.Now()
// 	headerAddEncoding, err := encryption.AddTwoCipherTensorNewMultiThread(publicKeys, combiningHeader, encodingCipherTensor)
// 	if err != nil {
// 		panic(err)
// 	}
// 	layerNorm1, err := maths.CiphertextTensorLayerNormReplaceVarianceMultiThread(publicKeys, headerAddEncoding, dashModelParam.LayerNormVectorR1, dashModelParam.LayerNormVectorB1, dashModelParam.LayerNormSqrtVariance1)
// 	if err != nil {
// 		panic(err)
// 	}
// 	elapsedTime = time.Since(startTime)
// 	fmt.Printf("LayerNorm-1 took %s to run.\n", elapsedTime)

// 	// 3.5. Feed Forward with ReLu
// 	startTime = time.Now()
// 	feedForwardRelu1, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, layerNorm1, dashModelParam.FeedForwardWeightMatrix1, dashModelParam.FeedForwardBiasVector1)
// 	if err != nil {
// 		panic(err)
// 	}
// 	// Relu
// 	feedForwardReluAppromate, err := maths.ApproximatePolynomialCipherTensorMultiThread(publicKeys, feedForwardRelu1, dashModelParam.ReluCoefficients, [2]float64{-50, 40})
// 	if err != nil {
// 		panic(err)
// 	}

// 	feedForwardRelu2, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, feedForwardReluAppromate, dashModelParam.FeedForwardWeightMatrix2, dashModelParam.FeedForwardBiasVector2)
// 	if err != nil {
// 		panic(err)
// 	}
// 	elapsedTime = time.Since(startTime)
// 	fmt.Printf("Feed Forward with ReLu took %s to run.\n", elapsedTime)

// 	// 3.6. Add and LayNorm-2
// 	startTime = time.Now()
// 	feedForwardAddlayerNorm1, err := encryption.AddTwoCipherTensorNewMultiThread(publicKeys, layerNorm1, feedForwardRelu2)
// 	if err != nil {
// 		panic(err)
// 	}
// 	// fmt.Printf("feedForwardAddlayerNorm1 scale :%f\n", feedForwardAddlayerNorm1.Ciphertexts[0].LogScale())
// 	layerNorm2, err := maths.CiphertextTensorLayerNormReplaceVarianceMultiThread(publicKeys, feedForwardAddlayerNorm1, dashModelParam.LayerNormVectorR2, dashModelParam.LayerNormVectorB2, dashModelParam.LayerNormSqrtVariance2)
// 	if err != nil {
// 		panic(err)
// 	}
// 	elapsedTime = time.Since(startTime)
// 	fmt.Printf("LayerNorm-2 took %s to run.\n", elapsedTime)

// 	// 3.7. Average Pooling
// 	// 3.8. Dense Classification
// 	startTime = time.Now()
// 	poolingAndClassification, err := maths.CiphertextTensorMultiplyClassificationAndPoolingMultiThread(publicKeys, layerNorm2, dashModelParam.ClassifierWeightMatrix, dashModelParam.ClassifierBiasVector)
// 	if err != nil {
// 		panic(err)
// 	}
// 	elapsedTime = time.Since(startTime)
// 	fmt.Printf("Pooling and dense classification took %s to run.\n", elapsedTime)

// 	return poolingAndClassification, nil
// }

func evalUnfoldDashformerWithBSGSMultiTread(publicKeys *encryption.PublicParametersKeys, secretKeys *encryption.SecretParametersKeys, exampleData [][][]float64,
	dashModelParam utils.DashformerModelParameters, coeff_dash coefficient.Coefficient_dash, coeff_QKV coefficient.Coefficient_QKV, coeff_sqmax coefficient.Coefficient_sqmax) (*encryption.CiphertextTensor, error) {
	//! NOTE: Here, secretKeys are just for debug and is not used for encrypted computation.
	// 2.2.加密example数据
	fmt.Println("Encrypting data ... ")
	startTime := time.Now()
	ciphertextTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, exampleData)
	if err != nil {
		panic(err)
	}
	elapsedTime := time.Since(startTime)
	fmt.Printf("Encrypting data ... takes %s\n", elapsedTime)
	// fmt.Printf("Data encryption completed, Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)
	// fmt.Printf("Ciphertext Tensor X0 Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())

	fmt.Println("Start computing with encrypted data")
	fmt.Printf("  ...")
	startTime = time.Now()
	startEncryptedComputation := time.Now()
	X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, err := maths.GenerateCipherTensorRot(publicKeys, ciphertextTensor, 7, 8)
	if err != nil {
		panic(err)
	}

	var concatenateHeader *encryption.CiphertextTensor
	// 3.3.Attention
	fmt.Printf("...")
	for i := 0; i < 4; i++ {

		multiAttentionHeader, err := maths.CipherTensorUnfoldX0ToAttentionWithBSGSMultiThread(publicKeys, ciphertextTensor, X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, coeff_sqmax.Item_1[i], coeff_sqmax.Item_2[i], coeff_sqmax.Item_3[i], coeff_sqmax.Item_4[i], coeff_QKV.A_V[i], coeff_QKV.Constant_V[i], 7, 8, dashModelParam.SoftMaxB[i], dashModelParam.SoftMaxC[i])
		if err != nil {
			panic(err)
		}
		// valueTensor, err := encryption.DecryptTensorValue(secretKeys, multiAttentionHeader)
		// if err != nil {
		// 	panic(err)
		// }
		// utils.PrintSliceInfo(valueTensor, "multiAttentionHeader")
		// fmt.Println(valueTensor[0])

		// fmt.Printf("Ciphertext Tensor multiAttentionHeader Level:%d\n", multiAttentionHeader.Ciphertexts[0].Level())
		// 3.3.5 concatenate header
		concatenateHeader, err = encryption.MergeAndAddCiphertextTensors(concatenateHeader, multiAttentionHeader)
		if err != nil {
			panic(err)
		}
	}

	// fmt.Printf("concatenateHeader Rows:%d, Cols:%d, Depth:%d\n", concatenateHeader.NumRows, concatenateHeader.NumCols, concatenateHeader.NumDepth)

	// 开始计算展开式
	// 1.计算rulu里面的内容
	fmt.Printf("...")
	cipherTensorHeaderBeforeRelu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coeff_dash.Head_before_relu, coeff_dash.Head_rear_relu)
	if err != nil {
		panic(err)
	}

	// valueTensor, err := encryption.DecryptTensorValue(secretKeys, cipherTensorHeaderBeforeRelu)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "cipherTensorHeaderBeforeRelu")
	// fmt.Println(valueTensor[0])
	fmt.Printf("...")
	cipherTensorX0BeforeRulu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coeff_dash.X0_before_relu, coeff_dash.X0_rear_relu)
	if err != nil {
		panic(err)
	}
	// cipherTensorBeforeRelu, err := encryption.AddTwoCipherTensorNewMultiThread(publicKeys, cipherTensorHeaderBeforeRelu, cipherTensorX0BeforeRulu)
	cipherTensorBeforeRelu, err := encryption.AddTwoCipherTensorNew(publicKeys, cipherTensorHeaderBeforeRelu, cipherTensorX0BeforeRulu)
	if err != nil {
		panic(err)
	}

	fmt.Printf("...\n")
	cipherTensorBeforeReluResult, err := maths.CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys, cipherTensorBeforeRelu, coeff_dash.Constant_Relu)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - before relu takes %s \n", elapsedTime)
	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorBeforeReluResult)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "cipherTensorBeforeReluResult")
	// fmt.Println(valueTensor[0])
	startTime = time.Now()
	// 2.1进行relu
	cipherTensorRelu, err := maths.ApproximatePolynomialCipherTensorMultiThread(publicKeys, cipherTensorBeforeReluResult, dashModelParam.ReluCoefficients, [2]float64{-50, 40})
	if err != nil {
		panic(err)
	}
	cipherTensorReluResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, cipherTensorRelu, coeff_dash.Relu_before, coeff_dash.Relu_rear)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - relu takes %s\n", elapsedTime)
	startTime = time.Now()
	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorReluResult)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "cipherTensorReluResult")
	// fmt.Println(valueTensor[0])

	// 2.2对HeadComplex进行计算
	cipherTensorHeadResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coeff_dash.Head_before, coeff_dash.Head_rear)
	if err != nil {
		panic(err)
	}

	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorHeadResult)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "cipherTensorHeadResult")
	// fmt.Println(valueTensor[0])

	// 2.3对X0进行计算
	cipherTensorX0Result, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coeff_dash.X0_before, coeff_dash.X0_rear)
	if err != nil {
		panic(err)
	}

	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorX0Result)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "cipherTensorX0Result")
	// fmt.Println(valueTensor[0])

	// 3.1将所有结果相加
	cipherTensorBeforePooling, err := encryption.AddThreeCipherTensorNewMultiThread(publicKeys, cipherTensorReluResult, cipherTensorHeadResult, cipherTensorX0Result)
	if err != nil {
		panic(err)
	}

	// 3.2进行pooling
	cipherTensorPoolingResult, err := maths.CipherTensorPoolingAndAddConstantMultiThread(publicKeys, cipherTensorBeforePooling, coeff_dash.Constant_Dash)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - after relu takes %s\n", elapsedTime)
	fmt.Printf("Encrypted computation takes %s\n", time.Since(startEncryptedComputation))
	return cipherTensorPoolingResult, nil
}

// func evalUnfoldDashformerMultiTread(publicKeys *encryption.PublicParametersKeys, secretKeys *encryption.SecretParametersKeys,
// 	exampleData [][][]float64, dashModelParam utils.DashformerModelParameters, coefficient coefficient.Coefficient_output) (*encryption.CiphertextTensor, error) {
// 	// 2.2.加密example数据
// 	ciphertextTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, exampleData)
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Printf("Encrypte data already, Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)
// 	// fmt.Printf("Ciphertext Tensor X0 Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())

// 	var concatenateHeader *encryption.CiphertextTensor
// 	// 3.3.Attention
// 	startTime := time.Now()
// 	for i := 0; i < 4; i++ {
// 		// 3.3.1. Compute Q,K,V
// 		multiAttentionQ, err := maths.CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, ciphertextTensor, coefficient.X0_rear_Q[i], coefficient.Constant_Q[i])
// 		if err != nil {
// 			panic(err)
// 		}

// 		multiAttentionK, err := maths.CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, ciphertextTensor, coefficient.X0_rear_K[i], coefficient.Constant_K[i])
// 		if err != nil {
// 			panic(err)
// 		}

// 		multiAttentionV, err := maths.CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, ciphertextTensor, coefficient.X0_rear_V[i], coefficient.Constant_V[i])
// 		if err != nil {
// 			panic(err)
// 		}

// 		// fmt.Printf("Ciphertext Tensor Q Level:%d\n", multiAttentionQ.Ciphertexts[0].Level())
// 		// fmt.Printf("Ciphertext Tensor K Level:%d\n", multiAttentionK.Ciphertexts[0].Level())
// 		// fmt.Printf("Ciphertext Tensor V Level:%d\n", multiAttentionV.Ciphertexts[0].Level())

// 		multiAttentionHeader, err := maths.CiphertextTensorQKVToAttentionWithBSGSMultiThread(publicKeys, multiAttentionQ, multiAttentionK, multiAttentionV, 7, 8, dashModelParam.SoftMaxB[i], dashModelParam.SoftMaxC[i])
// 		if err != nil {
// 			panic(err)
// 		}

// 		fmt.Printf("Ciphertext Tensor multiAttentionHeader Level:%d\n", multiAttentionHeader.Ciphertexts[0].Level())
// 		// 3.3.5 concatenate header
// 		concatenateHeader, err = encryption.MergeAndAddCiphertextTensors(concatenateHeader, multiAttentionHeader)
// 		if err != nil {
// 			panic(err)
// 		}
// 	}
// 	elapsedTime := time.Since(startTime)
// 	fmt.Printf("Attention took %s to run.\n", elapsedTime)

// 	// 开始计算展开式
// 	// 1.计算rulu里面的内容
// 	cipherTensorHeaderBeforeRelu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coefficient.Head_before_relu, coefficient.Head_rear_relu)
// 	if err != nil {
// 		panic(err)
// 	}
// 	cipherTensorX0BeforeRulu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coefficient.X0_before_relu, coefficient.X0_rear_relu)
// 	if err != nil {
// 		panic(err)
// 	}
// 	cipherTensorBeforeRelu, err := encryption.AddTwoCipherTensorNew(publicKeys, cipherTensorHeaderBeforeRelu, cipherTensorX0BeforeRulu)
// 	if err != nil {
// 		panic(err)
// 	}
// 	cipherTensorBeforeReluResult, err := maths.CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys, cipherTensorBeforeRelu, coefficient.Constant_Relu)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// valueTensor, err := encryption.DecryptTensorValue(secretKeys, cipherTensorBeforeReluResult)
// 	// if err != nil {
// 	// 	panic(err)
// 	// }
// 	// utils.PrintSliceInfo(valueTensor, "cipherTensorBeforeReluResult")
// 	// fmt.Println(valueTensor[0])

// 	// 2.1进行relu
// 	cipherTensorRelu, err := maths.ApproximatePolynomialCipherTensorMultiThread(publicKeys, cipherTensorBeforeReluResult, dashModelParam.ReluCoefficients, [2]float64{-50, 40})
// 	if err != nil {
// 		panic(err)
// 	}
// 	cipherTensorReluResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, cipherTensorRelu, coefficient.Relu_before, coefficient.Relu_rear)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorReluResult)
// 	// if err != nil {
// 	// 	panic(err)
// 	// }
// 	// utils.PrintSliceInfo(valueTensor, "cipherTensorReluResult")
// 	// fmt.Println(valueTensor[0])

// 	// 2.2对Head进行计算
// 	cipherTensorHeadResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coefficient.Head_before, coefficient.Head_rear)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorHeadResult)
// 	// if err != nil {
// 	// 	panic(err)
// 	// }
// 	// utils.PrintSliceInfo(valueTensor, "cipherTensorHeadResult")
// 	// fmt.Println(valueTensor[0])

// 	// 2.3对X0进行计算
// 	cipherTensorX0Result, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coefficient.X0_before, coefficient.X0_rear)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorX0Result)
// 	// if err != nil {
// 	// 	panic(err)
// 	// }
// 	// utils.PrintSliceInfo(valueTensor, "cipherTensorX0Result")
// 	// fmt.Println(valueTensor[0])

// 	// 3.1将所有结果相加
// 	cipherTensorBeforePooling, err := encryption.AddThreeCipherTensorNewMultiThread(publicKeys, cipherTensorReluResult, cipherTensorHeadResult, cipherTensorX0Result)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// 3.2进行pooling
// 	cipherTensorPoolingResult, err := maths.CipherTensorPoolingAndAddConstantMultiThread(publicKeys, cipherTensorBeforePooling, coefficient.Constant_Dash)
// 	if err != nil {
// 		panic(err)
// 	}

// 	return cipherTensorPoolingResult, nil

// }

func main() {
	startTime := time.Now()
	fmt.Println("Data reading ...")

	// 初始化配置
	examplePath, tokenizerPath, modelParamPath, outputPath := config.Init()
	// fmt.Println("Example Path:", examplePath)
	// fmt.Println("Tokenizer Path:", tokenizerPath)
	// fmt.Println("Model Parameter Path:", modelParamPath)
	// fmt.Println("Output Path:", outputPath)

	// 1. 读取文件数据
	// 1.1.读词向量文件，并解析成字典
	tokenizerDate, err := utils.ReadWordIndex(tokenizerPath)
	if err != nil {
		fmt.Println("Error reading dashformer_tokenizer.json file:", err)
		panic(err)
	}
	// fmt.Println(tokenizerDate)

	// 1.2.读输入示例，根据字典进行转换
	exampleData, err := utils.ReadExampleData(examplePath, tokenizerDate)
	if err != nil {
		log.Fatalf("Error reading example_AA_sequences.list file: %v", err)
		panic(err)
	}
	// utils.PrintSliceInfo(exampleData, "exampleDataTensor")

	// 1.3.读模型参数文件
	dashModelParam, err := utils.ReadModelParameterFile(modelParamPath)
	if err != nil {
		panic(err)
	}
	// 显示读取结果
	// dashModelParam.PrintDimensions()

	// 1.4.生成系数(unfold)
	coeff_dash, coeff_QKV, coeff_sqmax := coefficient.GenerateCoefficient(dashModelParam)

	// 2.1.生成加密参数
	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		panic(err)
	}

	// 进行密文计算
	// 调用 evalDashformer 函数并处理结果
	poolingAndClassification, err := evalUnfoldDashformerWithBSGSMultiTread(publicKeys, secretKeys, exampleData, dashModelParam, coeff_dash, coeff_QKV, coeff_sqmax)
	if err != nil {
		fmt.Println("Error in evalDashformer:", err)
	}

	fmt.Printf("  - ciphertexts now at level:%d\n", poolingAndClassification.Ciphertexts[0].Level())

	fmt.Println("Decrypting and writing the result ...")
	// 解密结果
	valueTensor, err := encryption.DecryptTensorValueMultiThread(secretKeys, poolingAndClassification)
	if err != nil {
		panic(err)
	}

	// 解密到文件中
	utils.WriteResultToFile(outputPath, valueTensor)

	elapsedTime := time.Since(startTime)
	fmt.Printf("Total running time is %s.\n", elapsedTime)

}

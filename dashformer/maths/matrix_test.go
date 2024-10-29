package maths

import (
	"dashformer/encryption"
	"dashformer/utils"
	"fmt"
	"testing"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func TestEncryptAndDecryptTensorValue(t *testing.T) {
	// 初始化参数和密钥对（假设已经定义好）

	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		panic(err)
	}

	plainTensorValue := [][][]float64{
		{
			{1.1, 2.2, 3.3},
			{4.4, 5.5, 6.6},
		},
		{
			{7.7, 8.8, 9.9},
			{10.10, 11.11, 12.12},
		},
	}
	plainMatrixValue1 := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	plainMatrixValue2 := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}

	plainVectorValue1 := []float64{10, 20, 30}
	plainVectorValue2 := []float64{0.1, 0.2, 0.3}

	ciphertextTensor, err := encryption.EncryptTensorValue(publicKeys, plainTensorValue)
	if err != nil {
		panic(err)
	}
	if len(ciphertextTensor.Ciphertexts) != len(plainTensorValue[0][0]) {
		t.Errorf("Expected %d ciphertexts, got %d", len(plainTensorValue[0][0]), len(ciphertextTensor.Ciphertexts))
	}
	fmt.Printf("Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)

	fmt.Printf("Ciphertext Tensor Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())
	fmt.Printf("Ciphertext Memery: %d MB\n", ciphertextTensor.Ciphertexts[0].BinarySize()/1048576)

	fmt.Printf("===============================================\n")
	fmt.Printf("Testing ciphertext tensor multiply plain matrix\n")
	fmt.Printf("===============================================\n")
	fmt.Printf("\n")
	// 进行明文×密文计算1
	newCiphertextTensor1, err := CiphertextTensorMultiplyPlaintextMatrix(publicKeys, ciphertextTensor, plainMatrixValue2)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor1 Level:%d\n", newCiphertextTensor1.Ciphertexts[0].Level())
	valueTensor, err := encryption.DecryptTensorValue(secretKeys, newCiphertextTensor1)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensor, "valueTensor")
	fmt.Println(valueTensor)

	// 进行明文+密文计算1
	fmt.Printf("==========================================\n")
	fmt.Printf("Testing ciphertext tensor add plain matrix\n")
	fmt.Printf("==========================================\n")
	fmt.Printf("\n")
	newCiphertextTensorAdd1, err := CiphertextTensorAddPlaintextMatrix(publicKeys, ciphertextTensor, plainMatrixValue1)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Add 1 Level:%d\n", newCiphertextTensorAdd1.Ciphertexts[0].Level())
	valueTensorAdd1, err := encryption.DecryptTensorValue(secretKeys, newCiphertextTensorAdd1)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensorAdd1, "valueTensorAdd")
	fmt.Println(valueTensorAdd1)

	// 进行密文张量×明文矩阵+偏置向量
	fmt.Printf("================================================================================\n")
	fmt.Printf("Testing ciphertext tensor multiply plain weight matrix and add plain bias vector\n")
	fmt.Printf("================================================================================\n")
	fmt.Printf("\n")
	newCiphertextTensorMulWeightAndAddBias, err := CiphertextTensorMultiplyWeightAndAddBias(publicKeys, ciphertextTensor, plainMatrixValue2, plainVectorValue1)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Multiply Weight Matrix And Add Bias Vector 1 Level:%d\n", newCiphertextTensorMulWeightAndAddBias.Ciphertexts[0].Degree())
	valueTensorMultiplyWeightAndAddBias, err := encryption.DecryptTensorValue(secretKeys, newCiphertextTensorMulWeightAndAddBias)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensorMultiplyWeightAndAddBias, "valueTensorMultiplyWeightAndAddBias")
	fmt.Println(valueTensorMultiplyWeightAndAddBias)

	// 进行旋转bycols
	fmt.Printf("==========================================\n")
	fmt.Printf("Testing ciphertext tensor Rotation by cols\n")
	fmt.Printf("==========================================\n")
	fmt.Printf("\n")
	newCiphertextTensorRot, err := CiphertextTensorRotationByColsNew(publicKeys, ciphertextTensor, 1, 10)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Rotation Level:%d\n", newCiphertextTensorRot.Ciphertexts[0].Level())
	valueTensorRot, err := encryption.DecryptTensorValue(secretKeys, newCiphertextTensorRot)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensorRot, "valueTensorRot")
	fmt.Println(valueTensorRot)

	// 进行密文×密文得到Halevi-Shoup编码
	fmt.Printf("====================================================================\n")
	fmt.Printf("Testing ciphertext tensor Multiply ciphertext tensor to Halevi-Shoup\n")
	fmt.Printf("====================================================================\n")
	fmt.Printf("\n")
	newCiphertextTensorToHaleviShoup, err := CiphertextTensorMultiplyCiphertextTensorToHalveiShoup(publicKeys, ciphertextTensor, ciphertextTensor, 10)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Rotation Level:%d\n", newCiphertextTensorToHaleviShoup.Ciphertexts[0].Level())
	valueTensorToHaleviShoup, err := encryption.DecryptTensorValue(secretKeys, newCiphertextTensorToHaleviShoup)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensorToHaleviShoup, "valueTensorToHaleviShoup")
	fmt.Println(valueTensorToHaleviShoup)

	// 进行密文H-S编码×密文cols编码
	fmt.Printf("===================================================================\n")
	fmt.Printf("Testing ciphertext tensor H-S Multiply ciphertext tensor to columns\n")
	fmt.Printf("===================================================================\n")
	fmt.Printf("\n")
	newCiphertextColumns, err := CiphertextTensorHSMultiplyCiphertextTensor(publicKeys, newCiphertextTensorToHaleviShoup, ciphertextTensor)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Rotation Level:%d\n", newCiphertextColumns.Ciphertexts[0].Level())
	valueTensorColumns, err := encryption.DecryptTensorValue(secretKeys, newCiphertextColumns)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensorColumns, "valueTensorColumns")
	fmt.Println(valueTensorColumns)

	// 返回均值和方差
	fmt.Printf("==========================================\n")
	fmt.Printf("Testing ciphertext tensor avg and variance\n")
	fmt.Printf("==========================================\n")
	fmt.Printf("\n")
	ctAvg, ctVar, err := CiphertextTensorReturnAvgAndVar(publicKeys, ciphertextTensor)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor ctAvg Level:%d\n", ctAvg.Level())
	fmt.Printf("Ciphertext Tensor ctVar Level:%d\n", ctVar.Level())
	encryption.DecryptVectorValue(secretKeys, ctAvg)
	encryption.DecryptVectorValue(secretKeys, ctVar)

	// 进行密文H-S编码×密文cols编码
	fmt.Printf("===================================\n")
	fmt.Printf("Testing ciphertext tensor LayerNorm\n")
	fmt.Printf("===================================\n")
	fmt.Printf("\n")
	fmt.Printf("Ciphertext Tensor Layernorm Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())
	sqrtLayerCoefficients1 := []float64{
		4.01447285e-01,
		-1.41122823e-02,
		3.37694161e-04,
		-4.54776425e-06,
		3.15551268e-08,
		-8.73491970e-11,
	}
	newCiphertextLayerNorm, err := CiphertextTensorLayerNorm(publicKeys, ciphertextTensor, plainVectorValue2, plainVectorValue1, sqrtLayerCoefficients1, [2]float64{20, 120})
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Layernorm Level:%d\n", newCiphertextLayerNorm.Ciphertexts[0].Level())
	valueTensorLayerNorm, err := encryption.DecryptTensorValue(secretKeys, newCiphertextLayerNorm)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensorLayerNorm, "valueTensorLayerNorm")
	fmt.Println(valueTensorLayerNorm)

	// 进行密文H-S编码×密文cols编码
	fmt.Printf("============================================\n")
	fmt.Printf("Testing ciphertext tensor LayerNormReduceMul\n")
	fmt.Printf("============================================\n")
	fmt.Printf("\n")
	fmt.Printf("Ciphertext Tensor Layernorm Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())
	newCiphertextLayerNormReduceMul, err := CiphertextTensorLayerNormReduceMul(publicKeys, ciphertextTensor, plainVectorValue2, plainVectorValue1, sqrtLayerCoefficients1, [2]float64{20, 120})
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Layernorm Level:%d\n", newCiphertextLayerNorm.Ciphertexts[0].Level())
	valueTensorLayerNormReduceMul, err := encryption.DecryptTensorValue(secretKeys, newCiphertextLayerNormReduceMul)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensorLayerNormReduceMul, "valueTensorLayerNormReduceMul")
	fmt.Println(valueTensorLayerNormReduceMul)

	// 进行测试pooling
	fmt.Printf("=====================================================\n")
	fmt.Printf("Testing ciphertext tensor pooling and classisfication\n")
	fmt.Printf("=====================================================\n")
	fmt.Printf("\n")
	fmt.Printf("Ciphertext Tensor Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())

	newCiphertextTensorPoolingAndClassification, err := CiphertextTensorMultiplyClassificationAndPooling(publicKeys, ciphertextTensor, plainMatrixValue2, plainVectorValue1)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor Classification And Pooling Level:%d\n", newCiphertextTensorPoolingAndClassification.Ciphertexts[0].Level())
	valueClassificationAndPooling, err := encryption.DecryptTensorValue(secretKeys, newCiphertextTensorPoolingAndClassification)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueClassificationAndPooling, "valueClassificationAndPooling")
	fmt.Println(valueClassificationAndPooling)

	// 进行测试pooling
	fmt.Printf("==================================\n")
	fmt.Printf("Testing ciphertext tensor Innersum\n")
	fmt.Printf("==================================\n")
	fmt.Printf("\n")

	ct := rlwe.NewCiphertext(publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
	err = publicKeys.Evaluator.InnerSum(ciphertextTensor.Ciphertexts[0], 2, 2, ct)
	if err != nil {
		panic(err)
	}

	// 解析密文
	pt := secretKeys.Decryptor.DecryptNew(ct)
	// Decodes the plaintext
	have := make([]float64, secretKeys.Params.MaxSlots())
	if err := secretKeys.Encoder.Decode(pt, have); err != nil {
		panic(err)
	}
	fmt.Println(have)
}

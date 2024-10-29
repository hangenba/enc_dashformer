package maths

import (
	"dashformer/encryption"
	"dashformer/utils"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

/*
 * CiphertextTensorMultiplyPlaintextMatrix
 * Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptMatrix Slice[][]
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) X ptMatrix(c,d) --> ctTensorNew(a,b,d)
 * 1CMul
 */
func CiphertextTensorMultiplyPlaintextMatrix(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, plainSlice [][]float64) (*encryption.CiphertextTensor, error) {

	// 返回维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherRows, cipherCols, cipherDepth)

	// 确定进行密文矩阵×明文矩阵的维数
	plainRows := len(plainSlice)
	plainCols := len(plainSlice[0])
	// fmt.Printf("Plaintext Matrix Rows:%d, Cols:%d\n", plainRows, plainCols)

	// 实际上，密文的depths必须等于明文的rows，才能继续进行运算；而明文的cols则是运算之后的depths
	if cipherDepth != plainRows {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the ciphertext tensor cannot multiply plaintext matrix: expected depth %d, got %d", plainRows, cipherDepth)
	}

	// 进行计算
	newCiphertexts := make([]*rlwe.Ciphertext, plainCols)
	for i := 0; i < plainCols; i++ {
		ct := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
		// ct.Scale = ciphertextTensor.Ciphertexts[0].Scale
		for j := 0; j < plainRows; j++ {
			// 创建一个明文向量
			ptMulNumSlice := make([]float64, cipherRows*cipherCols)
			for k := range ptMulNumSlice {
				ptMulNumSlice[k] = plainSlice[j][i]
			}

			// 进行乘法并相加
			ciphertextTensor.Ciphertexts[j].Scale = publicKeys.Params.DefaultScale()
			err := publicKeys.Evaluator.MulThenAdd(ciphertextTensor.Ciphertexts[j], ptMulNumSlice, ct)
			if err != nil {
				panic(err)
			}
		}

		if err := publicKeys.Evaluator.Rescale(ct, ct); err != nil {
			panic(err)
		}
		newCiphertexts[i] = ct
	}

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumCols:     cipherCols,
		NumRows:     cipherRows,
		NumDepth:    plainCols,
	}, nil
}

/*
 * CiphertextTensorAddPlaintextMatrix
 * Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptMatrix Slice[][]
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) + ptMatrix(b,c) --> ctTensorNew(a,b,c)
 */
func CiphertextTensorAddPlaintextMatrix(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, plainSlice [][]float64) (*encryption.CiphertextTensor, error) {
	// 返回维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherRows, cipherCols, cipherDepth)

	// 确定进行密文矩阵×明文矩阵的维数
	plainRows := len(plainSlice)
	plainCols := len(plainSlice[0])
	// fmt.Printf("Plaintext Matrix Rows:%d, Cols:%d\n", plainRows, plainCols)

	// 实际上，cipherCols必须等于plainRows; cipherDepth必须等于plainCols
	if cipherCols != plainRows || cipherDepth != plainCols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the ciphertext tensor cannot Add plaintext matrix: expected  (%d,%d), got (%d,%d)", plainRows, plainCols, cipherCols, cipherDepth)
	}

	newCiphertexts := make([]*rlwe.Ciphertext, cipherDepth)
	var err error
	for i := 0; i < cipherDepth; i++ {
		plainVector := make([]float64, cipherRows*cipherCols)
		for j := 0; j < cipherCols; j++ {
			for k := 0; k < cipherRows; k++ {
				plainVector[j+k*cipherCols] = plainSlice[j][i]
			}
		}
		newCiphertexts[i], err = publicKeys.Evaluator.AddNew(ciphertextTensor.Ciphertexts[i], plainVector)
		if err != nil {
			panic(err)
		}

	}

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumCols:     cipherCols,
		NumRows:     cipherRows,
		NumDepth:    cipherDepth,
	}, nil
}

/*
 * CiphertextTensorMultiplyWeightAndAddBias
 * Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptWeight Slice[][],ptBias Slice[]
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) X ptWeight(c,d) + ptBias(d) --> ctTensorNew(a,b,d)
 * 1CMul
 */
func CiphertextTensorMultiplyWeightAndAddBias(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, ptWeight [][]float64, ptBias []float64) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherRows, cipherCols, cipherDepth)

	// 返回明文权重矩阵维数
	weightRows := len(ptWeight)
	weightCols := len(ptWeight[0])
	// fmt.Printf("Plaintext Weight Matrix Rows:%d, Cols:%d\n", weightRows, weightCols)

	// 返回明文偏置维数
	biasLength := len(ptBias)
	// fmt.Printf("Plaintext Bias Vector Length:%d\n", biasLength)

	// 实际上，密文张量的depths必须等于权重矩阵的rows，才能继续进行运算；而权重矩阵的cols则是运算之后的depths需要等于biasLength
	if cipherDepth != weightRows && weightCols != biasLength {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the ciphertext tensor cannot multiply plaintext matrix: expected depth %d, got %d", weightRows, cipherDepth)
	}

	// Step 1.密文Tensor × 明文权重矩阵
	ciphertextTensorMulWeight, err := CiphertextTensorMultiplyPlaintextMatrix(publicKeys, ciphertextTensor, ptWeight)
	if err != nil {
		panic(err)
	}

	// Step 2. +偏置向量
	newCiphertexts := make([]*rlwe.Ciphertext, biasLength)
	for i := 0; i < biasLength; i++ {
		newCiphertexts[i], err = publicKeys.Evaluator.AddNew(ciphertextTensorMulWeight.Ciphertexts[i], ptBias[i])
		if err != nil {
			panic(err)
		}
	}

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumCols:     cipherCols,
		NumRows:     cipherRows,
		NumDepth:    biasLength,
	}, nil
}

/*
- CiphertextTensorMultiplyClassificationAndPooling
- Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptWeight Slice[][],ptBias Slice[],batchSize int64
- Output: CiphertextTensor,error
- Compute:
 1. ctTensor(a,b,c) X ptWeight(c,d)  --> ctTensorNew(a,b,d)
 2. ctTensorNew(a,b,d) pooling by batchsize b
 3. ctTensorNew(a,b,d) Add ptBias(d)

- 1CMul
*/
func CiphertextTensorMultiplyClassificationAndPooling(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, ptWeight [][]float64, ptBias []float64) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherRows, cipherCols, cipherDepth)

	// 返回明文权重矩阵维数
	weightRows := len(ptWeight)
	weightCols := len(ptWeight[0])
	// fmt.Printf("Plaintext Weight Matrix Rows:%d, Cols:%d\n", weightRows, weightCols)

	// 返回明文偏置维数
	biasLength := len(ptBias)
	// fmt.Printf("Plaintext Bias Vector Length:%d\n", biasLength)

	// 实际上，密文张量的depths必须等于权重矩阵的rows，才能继续进行运算；而权重矩阵的cols则是运算之后的depths需要等于biasLength
	if cipherDepth != weightRows && weightCols != biasLength {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the ciphertext tensor cannot multiply plaintext matrix: expected depth %d, got %d", weightRows, cipherDepth)
	}

	// Step 1. 密文Tensor × 明文权重矩阵
	ciphertextTensorMulWeight, err := CiphertextTensorMultiplyPlaintextMatrix(publicKeys, ciphertextTensor, ptWeight)
	if err != nil {
		panic(err)
	}

	// // 测试解密
	// valueTensor, err := encryption.DecryptTensorValue(secretKeys, ciphertextTensorMulWeight)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "ciphertextTensorMulWeight")
	// fmt.Println(valueTensor)

	// Step 2. 进行pooling
	// fmt.Println(ciphertextTensorMulWeight.NumCols)
	for i := 0; i < ciphertextTensorMulWeight.NumDepth; i++ {
		publicKeys.Evaluator.InnerSum(ciphertextTensorMulWeight.Ciphertexts[i], 1, ciphertextTensorMulWeight.NumCols, ciphertextTensorMulWeight.Ciphertexts[i])
	}

	// // 测试解密
	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, ciphertextTensorMulWeight)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "pooling")
	// fmt.Println(valueTensor)

	// Step 3. +偏置向量
	newCiphertexts := make([]*rlwe.Ciphertext, biasLength)
	for i := 0; i < biasLength; i++ {
		newCiphertexts[i], err = publicKeys.Evaluator.AddNew(ciphertextTensorMulWeight.Ciphertexts[i], ptBias[i])
		if err != nil {
			panic(err)
		}
	}

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumCols:     cipherCols,
		NumRows:     cipherRows,
		NumDepth:    biasLength,
	}, nil
}

/*
 * CiphertextTensorMultiplyCiphertextTensorToHalveiShoup
 * Input:  PublicParametersKeys,ctTensor1,ctTensor2 CiphertextTensor
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) X [ctTensor(a,b,c)^T --> ctTensorT(a,c,b)]--> ctTensorNew(a,b,b)
 * 1CMul+1Mul
 */
func CiphertextTensorMultiplyCiphertextTensorToHalveiShoup(publicKeys *encryption.PublicParametersKeys, cipherTensor1, cipherTensor2 *encryption.CiphertextTensor, baseSize float64) (*encryption.CiphertextTensor, error) {

	// 返回密文张量1维数
	cipherTensor1Rows := cipherTensor1.NumRows
	cipherTensor1Cols := cipherTensor1.NumCols
	cipherTensor1Depth := cipherTensor1.NumDepth
	// fmt.Printf("Ciphertext1 Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensor1Rows, cipherTensor1Cols, cipherTensor1Depth)

	// 返回密文张量2维数
	cipherTensor2Rows := cipherTensor2.NumRows
	cipherTensor2Cols := cipherTensor2.NumCols
	cipherTensor2Depth := cipherTensor2.NumDepth
	// fmt.Printf("Ciphertext2 Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensor2Rows, cipherTensor2Cols, cipherTensor2Depth)

	// 判断条件
	if cipherTensor1Cols != cipherTensor2Cols || cipherTensor1Rows != cipherTensor2Rows || cipherTensor1Depth != cipherTensor2Depth {
		return &encryption.CiphertextTensor{}, fmt.Errorf("can not multiply ctTensor1 and ctTensor2 transpose to Halevi-Shoup encodeing")
	}

	// 进行密文乘法
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensor1Cols)
	for i := 0; i < cipherTensor1Cols; i++ {
		ct := hefloat.NewCiphertext(*publicKeys.Params, cipherTensor1.Ciphertexts[0].Degree(), cipherTensor1.Ciphertexts[0].Level())
		// 进行旋转
		rotCiperTensor2, err := CiphertextTensorRotationByColsNew(publicKeys, cipherTensor2, i, baseSize)
		if err != nil {
			panic(err)
		}

		if rotCiperTensor2.NumDepth != cipherTensor1.NumDepth {
			fmt.Print("not equal")
		}

		// fmt.Println(i)
		for j := 0; j < cipherTensor1Depth; j++ {
			// fmt.Printf("stop here %d\n", j)
			// 进行乘法
			// fmt.Println(rotCiperTensor2.Ciphertexts[j])
			// fmt.Println(cipherTensor1.Ciphertexts[j])
			err = publicKeys.Evaluator.MulRelinThenAdd(rotCiperTensor2.Ciphertexts[j], cipherTensor1.Ciphertexts[j], ct)
			if err != nil {
				panic(err)
			}
		}
		// 进行rescale
		if err := publicKeys.Evaluator.Rescale(ct, ct); err != nil {
			panic(err)
		}
		newCiphertexts[i] = ct
	}
	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensor1Rows,
		NumCols:     cipherTensor1Cols,
		NumDepth:    cipherTensor1Cols,
	}, nil
}

/*
 * CiphertextTensorHSMultiplyCiphertextTensor
 * Input:  PublicParametersKeys,ctTensor1,ctTensor2 CiphertextTensor
 * Output: CiphertextTensor,error
 * Compute:
 	* ctTensor1 encoding by Halevi-Shoup
 	* ctTensor2 encoding by cols
 * 1CMul+1Mul
*/
func CiphertextTensorHSMultiplyCiphertextTensor(publicKeys *encryption.PublicParametersKeys, cipherTensor1, cipherTensor2 *encryption.CiphertextTensor) (*encryption.CiphertextTensor, error) {

	// 返回密文张量1维数
	// cipherTensor1Rows := cipherTensor1.NumRows
	cipherTensor1Cols := cipherTensor1.NumCols
	cipherTensor1Depth := cipherTensor1.NumDepth
	// fmt.Printf("Ciphertext1 Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensor1Rows, cipherTensor1Cols, cipherTensor1Depth)

	// 返回密文张量2维数
	cipherTensor2Rows := cipherTensor2.NumRows
	cipherTensor2Cols := cipherTensor2.NumCols
	cipherTensor2Depth := cipherTensor2.NumDepth
	// fmt.Printf("Ciphertext2 Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensor2Rows, cipherTensor2Cols, cipherTensor2Depth)

	// 判断条件H-S一定是一个方阵，tensor1.Depth等于tensor2.cols
	if cipherTensor1Cols != cipherTensor1Depth || cipherTensor1Depth != cipherTensor2Cols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("can not multiply ctTensor1HS and ctTensor2 to columns encodeing")
	}

	// 声明cipherTensor2Depth条密文
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensor2Depth)
	for j := 0; j < cipherTensor2Depth; j++ {
		newCiphertexts[j] = hefloat.NewCiphertext(*publicKeys.Params, cipherTensor1.Ciphertexts[0].Degree(), cipherTensor1.Ciphertexts[0].Level())
	}

	// 进行密文乘法
	for i := 0; i < cipherTensor2Cols; i++ {
		// 进行旋转
		rotCiperTensor2, err := CiphertextTensorRotationByColsNew(publicKeys, cipherTensor2, i, 1)
		if err != nil {
			panic(err)
		}

		for j := 0; j < cipherTensor2Depth; j++ {
			// fmt.Printf("stop here %d\n", j)
			// 进行乘法
			// fmt.Println(rotCiperTensor2.Ciphertexts[j])
			// fmt.Println(cipherTensor1.Ciphertexts[j])
			ct, err := publicKeys.Evaluator.MulRelinNew(rotCiperTensor2.Ciphertexts[j], cipherTensor1.Ciphertexts[i])
			if err != nil {
				panic(err)
			}

			publicKeys.Evaluator.Add(newCiphertexts[j], ct, newCiphertexts[j])
		}
	}

	// 进行rescale
	for j := 0; j < cipherTensor2Depth; j++ {
		publicKeys.Evaluator.Rescale(newCiphertexts[j], newCiphertexts[j])
	}
	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensor2Rows,
		NumCols:     cipherTensor2Cols,
		NumDepth:    cipherTensor2Depth,
	}, nil
}

/*
 * CiphertextTensorRotationByColsNew
 * Input:  PublicParametersKeys,ctTensor1 CiphertextTensor,rot int, base float64
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c)
 *  a × b  rot=1
 * |1 2 3|       |2 3 1|
 * |A B C|  -->  |B C A| ×base
 * |0 0 0|       |0 0 0|
 * 1CMul
 */
func CiphertextTensorRotationByColsNew(publicKeys *encryption.PublicParametersKeys, cipherTensor *encryption.CiphertextTensor, rotNumber int, baseSize float64) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherTensorRows := cipherTensor.NumRows
	cipherTensorCols := cipherTensor.NumCols
	cipherTensorDepth := cipherTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensorRows, cipherTensorCols, cipherTensorDepth)

	// 判断条件，如果rot大于cols，则说明旋转超出一行
	if rotNumber > cipherTensorCols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the rot size %d is too large, require <%d", rotNumber, cipherTensorCols)
	}

	// 先生成明文向量
	rotLeftVector := make([]float64, cipherTensorRows*cipherTensorCols)
	rotRightVector := make([]float64, cipherTensorRows*cipherTensorCols)
	for i := 0; i < cipherTensorRows; i++ {
		for j := 0; j < cipherTensorCols; j++ {
			if j < rotNumber {
				rotLeftVector[i*cipherTensorCols+j] = 0
				rotRightVector[i*cipherTensorCols+j] = baseSize
			} else {
				rotLeftVector[i*cipherTensorCols+j] = baseSize
				rotRightVector[i*cipherTensorCols+j] = 0
			}
		}
	}

	// fmt.Println(rotLeftVector)
	// fmt.Println(rotRightVector)
	// 进行计算
	Slots := publicKeys.Params.MaxSlots()
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensorDepth)
	for i := 0; i < cipherTensorDepth; i++ {
		ctLeft, err := publicKeys.Evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotLeftVector)
		if err != nil {
			panic(err)
		}
		ctRight, err := publicKeys.Evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotRightVector)
		if err != nil {
			panic(err)
		}

		// fmt.Printf("i-th:%d, rotLeft:%d, rotRight:%d\n", i, rotNumber, (rotNumber-cipherTensorCols+Slots)%Slots)
		publicKeys.Evaluator.Rotate(ctLeft, rotNumber, ctLeft)
		publicKeys.Evaluator.Rotate(ctRight, (rotNumber-cipherTensorCols+Slots)%Slots, ctRight)

		newCiphertexts[i], err = publicKeys.Evaluator.AddNew(ctLeft, ctRight)
		if err != nil {
			panic(err)
		}
		if err = publicKeys.Evaluator.Rescale(newCiphertexts[i], newCiphertexts[i]); err != nil {
			panic(err)
		}
	}

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensorRows,
		NumCols:     cipherTensorCols,
		NumDepth:    cipherTensorDepth,
	}, nil
}

/*
 * CiphertextTensorReturnAvgAndVar
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 */
func CiphertextTensorReturnAvgAndVar(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor) (*rlwe.Ciphertext, *rlwe.Ciphertext, error) {

	ct := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		err := publicKeys.Evaluator.Add(ct, ciphertextTensor.Ciphertexts[i], ct)
		if err != nil {
			panic(err)
		}
	}

	ctAvg, err := publicKeys.Evaluator.MulRelinNew(ct, float64(1/float64(ciphertextTensor.NumDepth)))
	if err != nil {
		panic(err)
	}

	publicKeys.Evaluator.Rescale(ctAvg, ctAvg)

	ctSqure := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		ctSub, err := publicKeys.Evaluator.SubNew(ciphertextTensor.Ciphertexts[i], ctAvg)
		if err != nil {
			panic(err)
		}

		err = publicKeys.Evaluator.MulRelinThenAdd(ctSub, ctSub, ctSqure)
		if err != nil {
			panic(err)
		}
	}
	publicKeys.Evaluator.Rescale(ctSqure, ctSqure)

	ctVar, err := publicKeys.Evaluator.MulRelinNew(ctSqure, float64(1/float64(ciphertextTensor.NumDepth)))
	if err != nil {
		panic(err)
	}
	publicKeys.Evaluator.Rescale(ctVar, ctVar)

	// fmt.Println(ctVar.Scale)
	// fmt.Println(ctAvg.Scale)

	return ctAvg, ctVar, nil
}

/*
 * CiphertextTensorLayerNorm
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor, layerNormR []float64, layerNormB []float64
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 */
func CiphertextTensorLayerNorm(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, layerNormR []float64, layerNormB []float64, coeff []float64, domain [2]float64) (*encryption.CiphertextTensor, error) {

	ctAvg, ctVar, err := CiphertextTensorReturnAvgAndVar(publicKeys, ciphertextTensor)
	if err != nil {
		panic(err)
	}

	if ciphertextTensor.NumDepth != len(layerNormR) || len(layerNormB) != len(layerNormR) {
		return &encryption.CiphertextTensor{}, fmt.Errorf("can not compute layernorm")
	}

	// 计算1/sqrt(ctVar)
	ctDownSqrtVar, err := ApproximatePolynomial(publicKeys, ctVar, coeff, domain)
	if err != nil {
		panic(err)
	}

	// 计算layerNorm
	newCiphertexts := make([]*rlwe.Ciphertext, ciphertextTensor.NumDepth)
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		newCiphertexts[i], err = publicKeys.Evaluator.SubNew(ciphertextTensor.Ciphertexts[i], ctAvg)
		if err != nil {
			panic(err)
		}
		publicKeys.Evaluator.Mul(newCiphertexts[i], layerNormR[i], newCiphertexts[i])
		err = publicKeys.Evaluator.Rescale(newCiphertexts[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}

		// 乘1/sqrt(ctVar)
		err = publicKeys.Evaluator.MulRelin(newCiphertexts[i], ctDownSqrtVar, newCiphertexts[i])
		if err != nil {
			panic(err)
		}
		err = publicKeys.Evaluator.Rescale(newCiphertexts[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}

		// 加B
		err = publicKeys.Evaluator.Add(newCiphertexts[i], layerNormB[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}
	}

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     ciphertextTensor.NumRows,
		NumCols:     ciphertextTensor.NumCols,
		NumDepth:    ciphertextTensor.NumDepth,
	}, nil
}

/*
 * CiphertextTensorLayerNormReduceMul
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor, layerNormR []float64, layerNormB []float64
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 */
func CiphertextTensorLayerNormReduceMul(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, layerNormR []float64, layerNormB []float64, coeff []float64, domain [2]float64) (*encryption.CiphertextTensor, error) {

	// compute sum
	ctSum := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		err := publicKeys.Evaluator.Add(ctSum, ciphertextTensor.Ciphertexts[i], ctSum)
		if err != nil {
			panic(err)
		}
	}

	ctVar := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		ctNX, err := publicKeys.Evaluator.MulNew(ciphertextTensor.Ciphertexts[i], ciphertextTensor.NumDepth)
		if err != nil {
			panic(err)
		}
		ctSub, err := publicKeys.Evaluator.SubNew(ctNX, ctSum)
		if err != nil {
			panic(err)
		}

		err = publicKeys.Evaluator.MulRelinThenAdd(ctSub, ctSub, ctVar)
		if err != nil {
			panic(err)
		}
	}
	publicKeys.Evaluator.Rescale(ctVar, ctVar)

	if ciphertextTensor.NumDepth != len(layerNormR) || len(layerNormB) != len(layerNormR) {
		return &encryption.CiphertextTensor{}, fmt.Errorf("can not compute layernorm")
	}

	// 计算1/sqrt(ctVar)
	ctDownSqrtVar, err := ApproximatePolynomial(publicKeys, ctVar, coeff, domain)
	if err != nil {
		panic(err)
	}

	// 计算layerNorm
	newCiphertexts := make([]*rlwe.Ciphertext, ciphertextTensor.NumDepth)
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		ctNX, err := publicKeys.Evaluator.MulNew(ciphertextTensor.Ciphertexts[i], ciphertextTensor.NumDepth)
		if err != nil {
			panic(err)
		}
		newCiphertexts[i], err = publicKeys.Evaluator.SubNew(ctNX, ctSum)
		if err != nil {
			panic(err)
		}

		publicKeys.Evaluator.Mul(newCiphertexts[i], layerNormR[i]*math.Sqrt(float64(ciphertextTensor.NumDepth)), newCiphertexts[i])
		err = publicKeys.Evaluator.Rescale(newCiphertexts[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}

		// 乘1/sqrt(ctVar)
		err = publicKeys.Evaluator.MulRelin(newCiphertexts[i], ctDownSqrtVar, newCiphertexts[i])
		if err != nil {
			panic(err)
		}
		err = publicKeys.Evaluator.Rescale(newCiphertexts[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}

		// 加B
		err = publicKeys.Evaluator.Add(newCiphertexts[i], layerNormB[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}
	}

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     ciphertextTensor.NumRows,
		NumCols:     ciphertextTensor.NumCols,
		NumDepth:    ciphertextTensor.NumDepth,
	}, nil
}

/*
 * CiphertextTensorLayerNormReplaceVariance
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor, layerNormR []float64, layerNormB []float64, varVector []float64
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 *
 */
func CiphertextTensorLayerNormReplaceVariance(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, layerNormR []float64, layerNormB []float64, varVector []float64) (*encryption.CiphertextTensor, error) {

	// compute sum
	ctSum := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		err := publicKeys.Evaluator.Add(ctSum, ciphertextTensor.Ciphertexts[i], ctSum)
		if err != nil {
			panic(err)
		}
	}

	if ciphertextTensor.NumDepth != len(layerNormR) || len(layerNormB) != len(layerNormR) {
		return &encryption.CiphertextTensor{}, fmt.Errorf("can not compute layernorm")
	}

	// 计算1/sqrt(ctVar)-->在此函数中，直接用明文varVector,因此这里直接跳过

	// 计算layerNorm，用明文varVector代替之后，先用r*varVector/n，再与密文相乘
	newCiphertexts := make([]*rlwe.Ciphertext, ciphertextTensor.NumDepth)
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		ctNX, err := publicKeys.Evaluator.MulNew(ciphertextTensor.Ciphertexts[i], ciphertextTensor.NumDepth)
		if err != nil {
			panic(err)
		}
		newCiphertexts[i], err = publicKeys.Evaluator.SubNew(ctNX, ctSum)
		if err != nil {
			panic(err)
		}

		// 对varVector进行放缩，也就是计算r*varVector/n
		ptVar := utils.ScaleVector(varVector, layerNormR[i]/float64(ciphertextTensor.NumDepth))
		// fmt.Println(layerNormR[i] / float64(ciphertextTensor.NumDepth))
		// 进行计算
		publicKeys.Evaluator.Mul(newCiphertexts[i], ptVar, newCiphertexts[i])
		err = publicKeys.Evaluator.Rescale(newCiphertexts[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}

		// 加B
		err = publicKeys.Evaluator.Add(newCiphertexts[i], layerNormB[i], newCiphertexts[i])
		if err != nil {
			panic(err)
		}
	}

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     ciphertextTensor.NumRows,
		NumCols:     ciphertextTensor.NumCols,
		NumDepth:    ciphertextTensor.NumDepth,
	}, nil
}

package maths

import (
	"dashformer/encryption"
	"dashformer/utils"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

/*
 * CiphertextTensorMultiplyPlaintextMatrixMultiThread
 * Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptMatrix Slice[][]
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) X ptMatrix(c,d) --> ctTensorNew(a,b,d)
 * 1CMul
 */
func CiphertextTensorMultiplyPlaintextMatrixMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, plainSlice [][]float64) (*encryption.CiphertextTensor, error) {

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

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	var mu sync.Mutex // 用于保护 newCiphertexts 的并发写入

	// 进行计算
	newCiphertexts := make([]*rlwe.Ciphertext, plainCols)
	for i := 0; i < plainCols; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// 复制evaluator
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 复制cipherTensor
			// newCiphertextTensor := ciphertextTensor.ShallowCopy()

			ct := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
			// ct.Scale = ciphertextTensor.Ciphertexts[0].Scale
			for j := 0; j < plainRows; j++ {
				// 创建一个明文向量
				ptMulNumSlice := make([]float64, cipherRows*cipherCols)
				for k := range ptMulNumSlice {
					ptMulNumSlice[k] = plainSlice[j][i]
				}

				// 进行乘法并相加
				// ciphertextTensor.Ciphertexts[j].Scale = publicKeys.Params.DefaultScale()
				err := evaluator.MulThenAdd(ciphertextTensor.Ciphertexts[j], ptMulNumSlice, ct)
				if err != nil {
					panic(err)
				}
			}

			if err := evaluator.Rescale(ct, ct); err != nil {
				panic(err)
			}
			// 并发安全地写入 newCiphertexts
			mu.Lock()
			newCiphertexts[i] = ct
			mu.Unlock()
		}(i)

	}
	// 等待所有 goroutine 完成
	wg.Wait()

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumCols:     cipherCols,
		NumRows:     cipherRows,
		NumDepth:    plainCols,
	}, nil
}

/*
 * CiphertextTensorAddPlaintextMatrixMultiThread
 * Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptMatrix Slice[][]
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) + ptMatrix(b,c) --> ctTensorNew(a,b,c)
 */
func CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, plainSlice [][]float64) (*encryption.CiphertextTensor, error) {
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

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// var mu sync.Mutex // 用于保护 newCiphertexts 的并发写入

	newCiphertexts := make([]*rlwe.Ciphertext, cipherDepth)
	var err error
	for i := 0; i < cipherDepth; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// 复制evaluator
			evaluator := publicKeys.Evaluator.ShallowCopy()

			//进行计算
			plainVector := make([]float64, cipherRows*cipherCols)
			for j := 0; j < cipherCols; j++ {
				for k := 0; k < cipherRows; k++ {
					plainVector[j+k*cipherCols] = plainSlice[j][i]
				}
			}

			newCiphertexts[i], err = evaluator.AddNew(ciphertextTensor.Ciphertexts[i], plainVector)
			if err != nil {
				panic(err)
			}
		}(i)
	}
	// 等待所有 goroutine 完成
	wg.Wait()

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumCols:     cipherCols,
		NumRows:     cipherRows,
		NumDepth:    cipherDepth,
	}, nil
}

/*
 * CiphertextTensorMultiplyWeightAndAddBiasMultiThread
 * Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptWeight Slice[][],ptBias Slice[]
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) X ptWeight(c,d) + ptBias(d) --> ctTensorNew(a,b,d)
 * 1CMul
 */
func CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, ptWeight [][]float64, ptBias []float64) (*encryption.CiphertextTensor, error) {

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
	ciphertextTensorMulWeight, err := CiphertextTensorMultiplyPlaintextMatrixMultiThread(publicKeys, ciphertextTensor, ptWeight)
	if err != nil {
		panic(err)
	}

	// Step 2. +偏置向量
	newCiphertexts := make([]*rlwe.Ciphertext, biasLength)
	for i := 0; i < biasLength; i++ {
		// ciphertextTensorMulWeight.Ciphertexts[i].Scale = publicKeys.Params.DefaultScale()
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
- CiphertextTensorMultiplyClassificationAndPoolingMultiThread
- Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptWeight Slice[][],ptBias Slice[],batchSize int64
- Output: CiphertextTensor,error
- Compute:
 1. ctTensor(a,b,c) X ptWeight(c,d)  --> ctTensorNew(a,b,d)
 2. ctTensorNew(a,b,d) pooling by batchsize b
 3. ctTensorNew(a,b,d) Add ptBias(d)

- 1CMul
*/
func CiphertextTensorMultiplyClassificationAndPoolingMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, ptWeight [][]float64, ptBias []float64) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth

	// fmt.Printf("before scale: %.5f", ptBias[0])

	// ptWeight = utils.ScalerMatrix(ptWeight, 1/600.)
	// ptBias = utils.ScaleVector(ptBias, 1/600.)

	// fmt.Printf("after scale: %.5f", ptBias[0])
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
	ciphertextTensorMulWeight, err := CiphertextTensorMultiplyPlaintextMatrixMultiThread(publicKeys, ciphertextTensor, ptWeight)
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
	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	for i := 0; i < ciphertextTensorMulWeight.NumDepth; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			evaluator.InnerSum(ciphertextTensorMulWeight.Ciphertexts[i], 1, ciphertextTensorMulWeight.NumCols, ciphertextTensorMulWeight.Ciphertexts[i])
		}(i)
	}
	wg.Wait()

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
		ciphertextTensor.Ciphertexts[i].Scale = publicKeys.Params.DefaultScale()
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
 * CiphertextTensorMultiplyCiphertextTensorToHalveiShoupMultiThread
 * Input:  PublicParametersKeys,ctTensor1,ctTensor2 CiphertextTensor
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c) X [ctTensor(a,b,c)^T --> ctTensorT(a,c,b)]--> ctTensorNew(a,b,b)
 * 1CMul+1Mul
 */
func CiphertextTensorMultiplyCiphertextTensorToHalveiShoupMultiThread(publicKeys *encryption.PublicParametersKeys, cipherTensor1, cipherTensor2 *encryption.CiphertextTensor, baseSize float64) (*encryption.CiphertextTensor, error) {

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

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	var mu sync.Mutex // 用于保护 newCiphertexts 的并发写入

	// 进行密文乘法
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensor1Cols)
	for i := 0; i < cipherTensor1Cols; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			ct := hefloat.NewCiphertext(*publicKeys.Params, cipherTensor1.Ciphertexts[0].Degree(), cipherTensor1.Ciphertexts[0].Level())
			// 进行旋转
			rotCiperTensor2, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, cipherTensor2, i, baseSize, publicKeys.Params.MaxSlots())
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
				err = evaluator.MulRelinThenAdd(rotCiperTensor2.Ciphertexts[j], cipherTensor1.Ciphertexts[j], ct)
				if err != nil {
					panic(err)
				}
			}
			// 进行rescale
			if err := evaluator.Rescale(ct, ct); err != nil {
				panic(err)
			}
			mu.Lock()
			newCiphertexts[i] = ct
			mu.Unlock()
		}(i)
	}
	wg.Wait()
	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensor1Rows,
		NumCols:     cipherTensor1Cols,
		NumDepth:    cipherTensor1Cols,
	}, nil
}

/*
 * CiphertextTensorHSMultiplyCiphertextTensorMultiThread
 * Input:  PublicParametersKeys,ctTensor1,ctTensor2 CiphertextTensor
 * Output: CiphertextTensor,error
 * Compute:
 	* ctTensor1 encoding by Halevi-Shoup
 	* ctTensor2 encoding by cols
 * 1CMul+1Mul
*/
func CiphertextTensorHSMultiplyCiphertextTensorMultiThread(publicKeys *encryption.PublicParametersKeys, cipherTensor1, cipherTensor2 *encryption.CiphertextTensor) (*encryption.CiphertextTensor, error) {

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

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	var mu sync.Mutex

	// 进行密文乘法
	for i := 0; i < cipherTensor2Cols; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 进行旋转
			rotCiperTensor2, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, cipherTensor2, i, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}

			localResults := make([]*rlwe.Ciphertext, cipherTensor2Depth)

			for j := 0; j < cipherTensor2Depth; j++ {
				// 进行乘法
				localResults[j], err = evaluator.MulRelinNew(rotCiperTensor2.Ciphertexts[j], cipherTensor1.Ciphertexts[i])
				if err != nil {
					// 处理错误
					fmt.Printf("Error in multiplication: %v\n", err)
					return
				}
				evaluator.Rescale(localResults[j], localResults[j])
			}

			// 合并局部结果
			mu.Lock()
			defer mu.Unlock()
			for j := 0; j < cipherTensor2Depth; j++ {
				evaluator.Add(newCiphertexts[j], localResults[j], newCiphertexts[j])
			}
		}(i)
	}
	wg.Wait()

	// // 进行rescale
	// for j := 0; j < cipherTensor2Depth; j++ {
	// 	publicKeys.Evaluator.Rescale(newCiphertexts[j], newCiphertexts[j])
	// }
	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensor2Rows,
		NumCols:     cipherTensor2Cols,
		NumDepth:    cipherTensor2Depth,
	}, nil
}

/*
 * CiphertextTensorQKVToAttentionWithBSGSMultiThread
 * Input:  PublicParametersKeys,Q,K,V CiphertextTensor
 * Output: CiphertextTensor,error
 * Compute: Q,K,V --> Attention result
 * 1CMul+1Mul
 */
func CiphertextTensorQKVToAttentionWithBSGSMultiThread(publicKeys *encryption.PublicParametersKeys, Q, K, V *encryption.CiphertextTensor, babyStep, giantStep int, b, c float64) (*encryption.CiphertextTensor, error) {

	// 返回密文张量1维数
	QCols := Q.NumCols
	QRows := Q.NumRows
	QDepth := Q.NumDepth
	// 返回密文张量2维数
	KRows := K.NumRows
	KCols := K.NumCols
	KDepth := K.NumDepth
	// 返回密文张量3维数
	VRows := V.NumRows
	VCols := V.NumCols
	VDepth := V.NumDepth
	// 判断QKV是否一样
	if QCols != KCols || QCols != VCols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("NumCols mismatch: Q.NumCols=%d, K.NumCols=%d, V.NumCols=%d", Q.NumCols, K.NumCols, V.NumCols)
	}
	if QDepth != KDepth || QDepth != VDepth {
		return &encryption.CiphertextTensor{}, fmt.Errorf("NumDepth mismatch: Q.NumDepth=%d, K.NumDepth=%d, V.NumDepth=%d", Q.NumDepth, K.NumDepth, V.NumDepth)
	}
	if QRows != KRows || QRows != VRows {
		return &encryption.CiphertextTensor{}, fmt.Errorf("NumRows mismatch: Q.NumRows=%d, K.NumRows=%d, V.NumRows=%d", Q.NumRows, K.NumRows, V.NumRows)
	}

	// 声明并初始化用于存储旋转结果的数组
	var QRotTensor = make([]*encryption.CiphertextTensor, giantStep)
	var KRotTensor = make([]*encryption.CiphertextTensor, babyStep)
	var VRotTensor = make([]*encryption.CiphertextTensor, babyStep)

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	var mu sync.Mutex

	// 生成旋转所有的步长
	for i := 0; i < giantStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 旋转Q
			// fmt.Printf("Q:%d,K:%d,V:%d\n", -i*babyStep, i, i)
			rotQ, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, Q, -i*babyStep, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			mu.Lock()
			QRotTensor[i] = rotQ
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	// 生成旋转所有的步长
	for i := 0; i < babyStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 旋转K
			rotK, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, K, i, 1/(math.Sqrt(32)*math.Sqrt(c)), publicKeys.Params.MaxSlots())
			// rotK, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, K, i, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			// 旋转V
			rotV, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, V, i, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			mu.Lock()
			KRotTensor[i] = rotK
			VRotTensor[i] = rotV
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	newCiphertexts := make([]*rlwe.Ciphertext, VDepth)
	for k := 0; k < VDepth; k++ {
		newCiphertexts[k] = hefloat.NewCiphertext(*publicKeys.Params, V.Ciphertexts[0].Degree(), V.Ciphertexts[0].Level())
	}

	// 进行BSGS To Attetion
	for i := 0; i < giantStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()

			localNewCiphertexts := make([]*rlwe.Ciphertext, VDepth)
			for k := 0; k < VDepth; k++ {
				localNewCiphertexts[k] = hefloat.NewCiphertext(*publicKeys.Params, V.Ciphertexts[0].Degree(), V.Ciphertexts[0].Level())
			}
			QKV := &encryption.CiphertextTensor{
				Ciphertexts: localNewCiphertexts,
				NumRows:     VRows,
				NumCols:     VCols,
				NumDepth:    VDepth,
			}
			for j := 0; j < babyStep; j++ {
				// 保证小于cols
				if (i*babyStep + j) < QCols {
					diagMatrix, err := CiphertextTensorMultiplyCiphertextTensorThenAdd(publicKeys.Params, evaluator, QRotTensor[i], KRotTensor[j])
					if err != nil {
						panic(err)
					}
					diagMatrixSoftMax, err := ApproximateSoftmaxCiphertext(evaluator, diagMatrix, b/math.Sqrt(c), 1)
					if err != nil {
						panic(err)
					}
					err = CiphertextTensorMultiplyCiphertextTensorAddToRes(publicKeys.Params, evaluator, diagMatrixSoftMax, VRotTensor[j], QKV)
					if err != nil {
						panic(err)
					}
				}
			}
			QKVRotKi, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, QKV, babyStep*i, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}

			// 合并局部结果
			mu.Lock()
			defer mu.Unlock()
			for k := 0; k < QKVRotKi.NumDepth; k++ {
				// newCiphertexts[k].Scale = publicKeys.Params.DefaultScale()
				// QKVRotKi.Ciphertexts[k].Scale = publicKeys.Params.DefaultScale()
				newCiphertexts[k], err = evaluator.AddNew(QKVRotKi.Ciphertexts[k], newCiphertexts[k])
				if err != nil {
					panic(err)
				}
			}
		}(i)
	}
	wg.Wait()
	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     QRows,
		NumCols:     QCols,
		NumDepth:    QDepth,
	}, nil
}

/*
 * CiphertextTensorQKVToAttentionWithBSGSMultiThread
 * Input:  PublicParametersKeys,Q,K,V CiphertextTensor
 * Output: *rlwe.Ciphertext,error
 * Compute: Q,K,V --> Attention result
 * 1CMul+1Mul
 */
func CiphertextTensorMultiplyCiphertextTensorThenAdd(param *hefloat.Parameters, evaluator *hefloat.Evaluator, cipherTensor1, cipherTensor2 *encryption.CiphertextTensor) (*rlwe.Ciphertext, error) {

	// 判断QKV是否一样
	if cipherTensor1.NumDepth != cipherTensor2.NumDepth {
		return &rlwe.Ciphertext{}, fmt.Errorf("NumCols mismatch: cipherTensor1.NumDepth=%d, cipherTensor2.NumDepth=%d", cipherTensor1.NumDepth, cipherTensor2.NumDepth)
	}

	ct := rlwe.NewCiphertext(param, cipherTensor1.Ciphertexts[0].Degree(), cipherTensor1.Ciphertexts[0].Level())
	// 进行BSGS To Attetion
	for i := 0; i < cipherTensor1.NumDepth; i++ {
		err := evaluator.MulRelinThenAdd(cipherTensor1.Ciphertexts[i], cipherTensor2.Ciphertexts[i], ct)
		if err != nil {
			panic(err)
		}

	}
	evaluator.Rescale(ct, ct)
	// 返回结果
	return ct, nil
}

/*
 * CiphertextTensorQKVToAttentionWithBSGSMultiThread
 * Input:  PublicParametersKeys,Q,K,V CiphertextTensor
 * Output: ,error
 * Compute: Q,K,V --> Attention result
 * 1CMul+1Mul
 */
func CiphertextTensorMultiplyCiphertextTensorAddToRes(param *hefloat.Parameters, evaluator *hefloat.Evaluator, ct1 *rlwe.Ciphertext, cipherTensor2 *encryption.CiphertextTensor, res *encryption.CiphertextTensor) error {

	// 进行BSGS To Attetion
	for i := 0; i < cipherTensor2.NumDepth; i++ {
		ctTmp, err := evaluator.MulRelinNew(cipherTensor2.Ciphertexts[i], ct1)
		if err != nil {
			panic(err)
		}
		evaluator.Rescale(ctTmp, ctTmp)
		//累加到res中
		ctTmp.Scale = param.DefaultScale()
		evaluator.Add(ctTmp, res.Ciphertexts[i], res.Ciphertexts[i])
	}

	// 返回结果
	return nil
}

/*
 * CiphertextTensorRotationByColsNewMultiThread
 * Input:  PublicParametersKeys,ctTensor1 CiphertextTensor,rot int, base float64
 * Output: CiphertextTensor,error
 * Compute:ctTensor(a,b,c)
 *  a × b  rot=1
 * |1 2 3|       |2 3 1|
 * |A B C|  -->  |B C A| ×base
 * |0 0 0|       |0 0 0|
 * 1CMul
 */
func CiphertextTensorRotationByColsNewMultiThread(evaluator *hefloat.Evaluator, cipherTensor *encryption.CiphertextTensor, rotNumber int, baseSize float64, Slots int) (*encryption.CiphertextTensor, error) {

	// Not rotate for rotNumber == 0, but also not multiply by baseSize
	if rotNumber == 0 {
		if math.Abs(baseSize-1.0) < 0.01 { // 9 calls
			return cipherTensor, nil
		}
	}

	// 返回密文张量维数
	cipherTensorRows := cipherTensor.NumRows
	cipherTensorCols := cipherTensor.NumCols
	cipherTensorDepth := cipherTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensorRows, cipherTensorCols, cipherTensorDepth)

	rotNumber = (rotNumber + cipherTensorCols) % cipherTensorCols
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
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensorDepth)
	for i := 0; i < cipherTensorDepth; i++ {
		ctLeft, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotLeftVector)
		if err != nil {
			panic(err)
		}
		ctRight, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotRightVector)
		if err != nil {
			panic(err)
		}

		// fmt.Printf("i-th:%d, rotLeft:%d, rotRight:%d\n", i, rotNumber, (rotNumber-cipherTensorCols+Slots)%Slots)
		evaluator.Rotate(ctLeft, rotNumber, ctLeft)
		evaluator.Rotate(ctRight, (rotNumber-cipherTensorCols+Slots)%Slots, ctRight)

		newCiphertexts[i], err = evaluator.AddNew(ctLeft, ctRight)
		if err != nil {
			panic(err)
		}
		if err = evaluator.Rescale(newCiphertexts[i], newCiphertexts[i]); err != nil {
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

func CiphertextTensorRotationByColsNewMultiThread_bak(evaluator *hefloat.Evaluator, cipherTensor *encryption.CiphertextTensor, rotNumber int, baseSize float64, Slots int) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherTensorRows := cipherTensor.NumRows
	cipherTensorCols := cipherTensor.NumCols
	cipherTensorDepth := cipherTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensorRows, cipherTensorCols, cipherTensorDepth)

	rotNumber = (rotNumber + cipherTensorCols) % cipherTensorCols
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
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensorDepth)
	for i := 0; i < cipherTensorDepth; i++ {
		ctLeft, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotLeftVector)
		if err != nil {
			panic(err)
		}
		ctRight, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotRightVector)
		if err != nil {
			panic(err)
		}

		// fmt.Printf("i-th:%d, rotLeft:%d, rotRight:%d\n", i, rotNumber, (rotNumber-cipherTensorCols+Slots)%Slots)
		evaluator.Rotate(ctLeft, rotNumber, ctLeft)
		evaluator.Rotate(ctRight, (rotNumber-cipherTensorCols+Slots)%Slots, ctRight)

		newCiphertexts[i], err = evaluator.AddNew(ctLeft, ctRight)
		if err != nil {
			panic(err)
		}
		if err = evaluator.Rescale(newCiphertexts[i], newCiphertexts[i]); err != nil {
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
 * CiphertextTensorLayerNormReplaceVarianceMultiThread
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor, layerNormR []float64, layerNormB []float64, varVector []float64
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 *
 */
func CiphertextTensorLayerNormReplaceVarianceMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, layerNormR []float64, layerNormB []float64, varVector []float64) (*encryption.CiphertextTensor, error) {

	// compute sum
	ctSum := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		err := publicKeys.Evaluator.Add(ctSum, ciphertextTensor.Ciphertexts[i], ctSum)
		if err != nil {
			panic(err)
		}
	}

	// 将向量重复ciphertextTensor.NumRows次
	varVector, err := utils.ReaptVector(varVector, ciphertextTensor.NumRows)
	if err != nil {
		panic(err)
	}

	if ciphertextTensor.NumDepth != len(layerNormR) || len(layerNormB) != len(layerNormR) {
		return &encryption.CiphertextTensor{}, fmt.Errorf("can not compute layernorm")
	}

	// 计算1/sqrt(ctVar)-->在此函数中，直接用明文varVector,因此这里直接跳过
	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	var mu sync.Mutex

	// 计算layerNorm，用明文varVector代替之后，先用r*varVector/n，再与密文相乘
	newCiphertexts := make([]*rlwe.Ciphertext, ciphertextTensor.NumDepth)
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()

			ctNX, err := evaluator.MulNew(ciphertextTensor.Ciphertexts[i], ciphertextTensor.NumDepth)
			if err != nil {
				panic(err)
			}
			ctTmp, err := evaluator.SubNew(ctNX, ctSum)
			if err != nil {
				panic(err)
			}

			// 对varVector进行放缩，也就是计算r*varVector/n
			mu.Lock()
			ptVar := utils.ScaleVector(varVector, layerNormR[i]/float64(ciphertextTensor.NumDepth))
			mu.Unlock()

			// fmt.Println(layerNormR[i] / float64(ciphertextTensor.NumDepth))
			// 进行计算
			evaluator.Mul(ctTmp, ptVar, ctTmp)
			err = evaluator.Rescale(ctTmp, ctTmp)
			if err != nil {
				panic(err)
			}

			// 加B
			err = evaluator.Add(ctTmp, layerNormB[i], ctTmp)
			if err != nil {
				panic(err)
			}

			mu.Lock()
			newCiphertexts[i] = ctTmp
			mu.Unlock()

		}(i)

	}
	wg.Wait()

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     ciphertextTensor.NumRows,
		NumCols:     ciphertextTensor.NumCols,
		NumDepth:    ciphertextTensor.NumDepth,
	}, nil
}

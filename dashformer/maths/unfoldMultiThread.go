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
 * PlainVecMulCipherTensorMulPlainMatMultiThread
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor, beforeVec []float64, rearMat [][]float64
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 * compute: diag(beforeVec) X ctTensor X rearMat
 */
func PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, beforeVec []float64, rearMat [][]float64) (*encryption.CiphertextTensor, error) {

	// 将向量重复ciphertextTensor.NumRows次
	beforeVec, err := utils.ReaptVector(beforeVec, ciphertextTensor.NumRows)
	if err != nil {
		panic(err)
	}

	// 返回维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherRows, cipherCols, cipherDepth)

	// 确定进行密文矩阵×明文矩阵的维数
	plainRows := len(rearMat)
	plainCols := len(rearMat[0])
	// fmt.Printf("Plaintext Matrix Rows:%d, Cols:%d\n", plainRows, plainCols)

	// 实际上，密文的depths必须等于明文的rows，才能继续进行运算；而明文的cols则是运算之后的depths
	if cipherDepth != plainRows {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the ciphertext tensor cannot multiply plaintext matrix: expected depth %d, got %d", plainRows, cipherDepth)
	}

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// var mu sync.Mutex // 用于保护 newCiphertexts 的并发写入

	// 进行计算
	newCiphertexts := make([]*rlwe.Ciphertext, plainCols)
	for i := 0; i < plainCols; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// 复制evaluator
			evaluator := publicKeys.Evaluator.ShallowCopy()

			ct := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
			// ct.Scale = ciphertextTensor.Ciphertexts[0].Scale
			for j := 0; j < plainRows; j++ {
				// 创建一个明文向量
				ptMulNumSlice := make([]float64, cipherRows*cipherCols)
				for k := range ptMulNumSlice {
					ptMulNumSlice[k] = rearMat[j][i]
				}

				plainVecScale := utils.ScaleVector(beforeVec, rearMat[j][i])

				// 进行乘法并相加
				ciphertextTensor.Ciphertexts[j].Scale = publicKeys.Params.DefaultScale()
				err := evaluator.MulThenAdd(ciphertextTensor.Ciphertexts[j], plainVecScale, ct)
				if err != nil {
					panic(err)
				}
			}

			if err := evaluator.Rescale(ct, ct); err != nil {
				panic(err)
			}
			// 并发安全地写入 newCiphertexts
			// mu.Lock()
			newCiphertexts[i] = ct
			// mu.Unlock()
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
- CipherTensorPoolingAndAddConstantMultiThread
- Input:  PublicParametersKeys,ctTensor CiphertextTensor,ptBias []float64
- Output: CiphertextTensor,error
- 1CMul
*/
func CipherTensorPoolingAndAddConstantMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, ptBias []float64) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth

	if cipherDepth != len(ptBias) {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the ciphertext tensor cannot multiply plaintext matrix: expected depth %d, got %d", len(ptBias), cipherDepth)
	}

	// Step 2. 进行pooling
	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup

	newCiphertexts := make([]*rlwe.Ciphertext, cipherDepth)
	var err error
	for i := 0; i < cipherDepth; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()

			evaluator.InnerSum(ciphertextTensor.Ciphertexts[i], 1, ciphertextTensor.NumCols, ciphertextTensor.Ciphertexts[i])
			ciphertextTensor.Ciphertexts[i].Scale = publicKeys.Params.DefaultScale()
			newCiphertexts[i], err = evaluator.AddNew(ciphertextTensor.Ciphertexts[i], ptBias[i])
			if err != nil {
				panic(err)
			}
		}(i)
	}
	wg.Wait()

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumCols:     cipherCols,
		NumRows:     cipherRows,
		NumDepth:    cipherDepth,
	}, nil
}

/*
 * CipherTensorMulPlainMatAndAddPlainMatMultiThread
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor, rearMat [][]float64, addMat [][]float64
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 * Compute: ciphertextTensor X rearMat + addMat
 */
func CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, rearMat [][]float64, addMat [][]float64) (*encryption.CiphertextTensor, error) {

	// 返回维数
	cipherRows := ciphertextTensor.NumRows
	cipherCols := ciphertextTensor.NumCols
	cipherDepth := ciphertextTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherRows, cipherCols, cipherDepth)

	// 确定进行密文矩阵×明文矩阵的维数
	plainRows := len(rearMat)
	plainCols := len(rearMat[0])
	// fmt.Printf("Plaintext MulMatrix Rows:%d, Cols:%d\n", plainRows, plainCols)

	addRows := len(addMat)
	addCols := len(addMat[0])
	// fmt.Printf("Plaintext AddMatrix Rows:%d, Cols:%d\n", addRows, addCols)

	addVector := make([][]float64, addCols)
	for i := 0; i < addCols; i++ {
		vec := make([]float64, addRows)
		for j := 0; j < addRows; j++ {
			vec[j] = addMat[j][i]
		}
		vec, err := utils.ReaptVector(vec, cipherRows)
		if err != nil {
			panic(err)
		}
		addVector[i] = vec
	}

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

			ct := hefloat.NewCiphertext(*publicKeys.Params, ciphertextTensor.Ciphertexts[0].Degree(), ciphertextTensor.Ciphertexts[0].Level())
			// ct.Scale = ciphertextTensor.Ciphertexts[0].Scale
			for j := 0; j < plainRows; j++ {
				// 创建一个明文向量
				ptMulNumSlice := make([]float64, cipherRows*cipherCols)
				for k := range ptMulNumSlice {
					ptMulNumSlice[k] = rearMat[j][i]
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
			if err := evaluator.Add(ct, addVector[i], ct); err != nil {
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
 * CipherTensorToQKTWithBSGSMultiThread
 * Input:  PublicParametersKeys, ctTensor *CiphertextTensor, rearMat [][]float64, addMat [][]float64
 * Output: *rlwe.Ciphertext, *rlwe.Ciphertext, error
 * Compute: ciphertextTensor X rearMat + addMat
 */
func CipherTensorUnfoldX0ToAttentionWithBSGSMultiThread(publicKeys *encryption.PublicParametersKeys, X0 *encryption.CiphertextTensor, X0Tensor, X0TRotTensor, X0RotTensorLeft, X0RotTensorRight []*encryption.CiphertextTensor, WQWKT, WQBKT, BQWKT, BQBKT [][]float64,
	X0_rear_V, Constant_V [][]float64, babyStep, giantStep int, b, c float64) (*encryption.CiphertextTensor, error) {

	V, err := CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, X0, X0_rear_V, Constant_V)
	if err != nil {
		panic(err)
	}

	VCols := V.NumCols
	VRows := V.NumRows
	VDepth := V.NumDepth

	// 声明并初始化用于存储旋转结果的数组
	// var X0RotTensorLeft = make([]*encryption.CiphertextTensor, giantStep)
	// var X0RotTensorRight = make([]*encryption.CiphertextTensor, giantStep)
	// var X0TRotTensor = make([]*encryption.CiphertextTensor, babyStep)
	var VRotTensor = make([]*encryption.CiphertextTensor, babyStep)

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex

	// // 生成旋转所有的步长
	// for i := 0; i < giantStep; i++ {
	// 	wg.Add(1)
	// 	go func(i int) {
	// 		defer wg.Done()
	// 		evaluator := publicKeys.Evaluator.ShallowCopy()
	// 		// 旋转Q
	// 		// fmt.Printf("Q:%d,K:%d,V:%d\n", -i*babyStep, i, i)
	// 		rotX0Left, rotX0Right, err := CipherTensorRotationByColsNotAddMultiThread(evaluator, X0, -i*babyStep, 1, publicKeys.Params.MaxSlots())
	// 		if err != nil {
	// 			panic(err)
	// 		}
	// 		mu.Lock()
	// 		X0RotTensorLeft[i] = rotX0Left
	// 		X0RotTensorRight[i] = rotX0Right
	// 		mu.Unlock()
	// 	}(i)
	// }
	// wg.Wait()

	// 生成旋转所有的步长
	for i := 0; i < babyStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 旋转K

			// rotX0T, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, X0, i, 1, publicKeys.Params.MaxSlots())
			// if err != nil {
			// 	panic(err)
			// }
			// 旋转V
			rotV, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, V, i, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			// mu.Lock()
			// X0TRotTensor[i] = rotX0T
			VRotTensor[i] = rotV
			// mu.Unlock()
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
			X0TensorWQWKT, err := CipherTensorMulPlainMatWithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, X0RotTensorLeft[i], X0RotTensorRight[i], WQWKT, -i*babyStep, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			X0TensorWQWKTAdd, err := CiphertextTensorAddPlaintextMatrixWithEvaluator(evaluator, X0TensorWQWKT, rotateMatrixColumns(BQWKT, -i*babyStep))
			if err != nil {
				panic(err)
			}
			for j := 0; j < babyStep; j++ {
				// 保证小于cols
				if (i*babyStep + j) < VCols {
					// 得到diagMatrix
					diagMatrix_1, err := CiphertextTensorMultiplyCiphertextTensorThenAdd(publicKeys.Params, evaluator, X0TensorWQWKTAdd, X0TRotTensor[j])
					if err != nil {
						panic(err)
					}
					digaMatrix_2, err := CiphertextTensorMultiplyPlainMatThenAdd(publicKeys.Params, evaluator, X0Tensor[i], rotateMatrixRows(WQBKT, j))
					if err != nil {
						panic(err)
					}
					// diagMatrix_3, err := PlainMatMultiplyCiphertextTensorThenAdd(publicKeys.Params, evaluator, rotateMatrixColumns(BQWKT, -i*babyStep), X0TRotTensor[j])
					// if err != nil {
					// 	panic(err)
					// }
					diagMatrix, err := evaluator.AddNew(diagMatrix_1, digaMatrix_2)
					if err != nil {
						panic(err)
					}
					// err = evaluator.Add(diagMatrix, diagMatrix_3, diagMatrix)
					// if err != nil {
					// 	panic(err)
					// }
					// 获得明文BQBKT项的第i条对角线
					diagPlain, err := GetDiagRotVector(BQBKT, i*babyStep+j, -i*babyStep)
					if err != nil {
						panic(err)
					}
					diagPlain, err = utils.ReaptVector(diagPlain, VRows)
					if err != nil {
						panic(err)
					}
					err = evaluator.Add(diagMatrix, diagPlain, diagMatrix)
					if err != nil {
						panic(err)
					}

					// softmax
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
			// mu.Lock()
			// defer mu.Unlock()
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
		NumRows:     VRows,
		NumCols:     VCols,
		NumDepth:    VDepth,
	}, nil
}

func CipherTensorUnfoldX0ToAttentionWithBSGSMultiThread_bak(publicKeys *encryption.PublicParametersKeys, X0 *encryption.CiphertextTensor, X0Tensor, X0TRotTensor, X0RotTensorLeft, X0RotTensorRight []*encryption.CiphertextTensor, WQWKT, WQBKT, BQWKT, BQBKT [][]float64,
	X0_rear_V, Constant_V [][]float64, babyStep, giantStep int, b, c float64) (*encryption.CiphertextTensor, error) {

	V, err := CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, X0, X0_rear_V, Constant_V)
	if err != nil {
		panic(err)
	}

	VCols := V.NumCols
	VRows := V.NumRows
	VDepth := V.NumDepth

	// 声明并初始化用于存储旋转结果的数组
	// var X0RotTensorLeft = make([]*encryption.CiphertextTensor, giantStep)
	// var X0RotTensorRight = make([]*encryption.CiphertextTensor, giantStep)
	// var X0TRotTensor = make([]*encryption.CiphertextTensor, babyStep)
	var VRotTensor = make([]*encryption.CiphertextTensor, babyStep)

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex

	// // 生成旋转所有的步长
	// for i := 0; i < giantStep; i++ {
	// 	wg.Add(1)
	// 	go func(i int) {
	// 		defer wg.Done()
	// 		evaluator := publicKeys.Evaluator.ShallowCopy()
	// 		// 旋转Q
	// 		// fmt.Printf("Q:%d,K:%d,V:%d\n", -i*babyStep, i, i)
	// 		rotX0Left, rotX0Right, err := CipherTensorRotationByColsNotAddMultiThread(evaluator, X0, -i*babyStep, 1, publicKeys.Params.MaxSlots())
	// 		if err != nil {
	// 			panic(err)
	// 		}
	// 		mu.Lock()
	// 		X0RotTensorLeft[i] = rotX0Left
	// 		X0RotTensorRight[i] = rotX0Right
	// 		mu.Unlock()
	// 	}(i)
	// }
	// wg.Wait()

	// 生成旋转所有的步长
	for i := 0; i < babyStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 旋转K

			// rotX0T, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, X0, i, 1, publicKeys.Params.MaxSlots())
			// if err != nil {
			// 	panic(err)
			// }
			// 旋转V
			rotV, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, V, i, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			// mu.Lock()
			// X0TRotTensor[i] = rotX0T
			VRotTensor[i] = rotV
			// mu.Unlock()
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
			X0TensorWQWKT, err := CipherTensorMulPlainMatWithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, X0RotTensorLeft[i], X0RotTensorRight[i], WQWKT, -i*babyStep, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			// X0Tensor, err := CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, X0RotTensorLeft[i], X0RotTensorRight[i], -i*babyStep, publicKeys.Params.MaxSlots())
			// if err != nil {
			// 	panic(err)
			// }
			for j := 0; j < babyStep; j++ {
				// 保证小于cols
				if (i*babyStep + j) < VCols {
					// 得到diagMatrix
					X0TensorWQWKTAdd, err := CiphertextTensorAddPlaintextMatrixWithEvaluator(evaluator, X0TensorWQWKT, rotateMatrixColumns(BQWKT, -i*babyStep))
					if err != nil {
						panic(err)
					}
					diagMatrix_1, err := CiphertextTensorMultiplyCiphertextTensorThenAdd(publicKeys.Params, evaluator, X0TensorWQWKTAdd, X0TRotTensor[j])
					if err != nil {
						panic(err)
					}
					digaMatrix_2, err := CiphertextTensorMultiplyPlainMatThenAdd(publicKeys.Params, evaluator, X0Tensor[i], rotateMatrixRows(WQBKT, j))
					if err != nil {
						panic(err)
					}
					// diagMatrix_3, err := PlainMatMultiplyCiphertextTensorThenAdd(publicKeys.Params, evaluator, rotateMatrixColumns(BQWKT, -i*babyStep), X0TRotTensor[j])
					// if err != nil {
					// 	panic(err)
					// }
					diagMatrix, err := evaluator.AddNew(diagMatrix_1, digaMatrix_2)
					if err != nil {
						panic(err)
					}
					// err = evaluator.Add(diagMatrix, diagMatrix_3, diagMatrix)
					// if err != nil {
					// 	panic(err)
					// }
					// 获得明文BQBKT项的第i条对角线
					diagPlain, err := GetDiagRotVector(BQBKT, i*babyStep+j, -i*babyStep)
					if err != nil {
						panic(err)
					}
					diagPlain, err = utils.ReaptVector(diagPlain, VRows)
					if err != nil {
						panic(err)
					}
					err = evaluator.Add(diagMatrix, diagPlain, diagMatrix)
					if err != nil {
						panic(err)
					}

					// softmax
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
			// mu.Lock()
			// defer mu.Unlock()
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
		NumRows:     VRows,
		NumCols:     VCols,
		NumDepth:    VDepth,
	}, nil
}

/*
 * CiphertextTensorRotationByColsNotAddMultiThread
 * Input:  PublicParametersKeys,ctTensor1 CiphertextTensor,rot int
 * Output: CiphertextTensorLeft, CiphertextTensorRight, error
 * Compute:ctTensor(a,b,c)  ----   [A,B,C,D,E,|0,0,0]
 *         CiphertextTensorLeft    [C,D,E,0,0,|0,A,B]
 *         CiphertextTensorRight   [0,0,0,A,B,|C,D,E]
 */
func CipherTensorRotationByColsNotAddMultiThread(evaluator *hefloat.Evaluator, cipherTensor *encryption.CiphertextTensor, rotNumber int, baseSzie float64, Slots int) (*encryption.CiphertextTensor, *encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherTensorRows := cipherTensor.NumRows
	cipherTensorCols := cipherTensor.NumCols
	cipherTensorDepth := cipherTensor.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensorRows, cipherTensorCols, cipherTensorDepth)

	rotNumber = (rotNumber + cipherTensorCols) % cipherTensorCols
	// 判断条件，如果rot大于cols，则说明旋转超出一行
	if rotNumber > cipherTensorCols {
		return &encryption.CiphertextTensor{}, &encryption.CiphertextTensor{}, fmt.Errorf("the rot size %d is too large, require <%d", rotNumber, cipherTensorCols)
	}

	// // 先生成明文向量
	// rotLeftVector := make([]float64, cipherTensorRows*cipherTensorCols)
	// rotRightVector := make([]float64, cipherTensorRows*cipherTensorCols)
	// for i := 0; i < cipherTensorRows; i++ {
	// 	for j := 0; j < cipherTensorCols; j++ {
	// 		if j < rotNumber {
	// 			rotLeftVector[i*cipherTensorCols+j] = 0
	// 			rotRightVector[i*cipherTensorCols+j] = baseSize
	// 		} else {
	// 			rotLeftVector[i*cipherTensorCols+j] = baseSize
	// 			rotRightVector[i*cipherTensorCols+j] = 0
	// 		}
	// 	}
	// }

	// fmt.Println(rotLeftVector)
	// fmt.Println(rotRightVector)
	// fmt.Println(rotNumber)
	// 进行计算
	newCiphertextsLeft := make([]*rlwe.Ciphertext, cipherTensorDepth)
	newCiphertextsRight := make([]*rlwe.Ciphertext, cipherTensorDepth)
	for i := 0; i < cipherTensorDepth; i++ {
		// ctLeft, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotLeftVector)
		// if err != nil {
		// 	panic(err)
		// }
		// ctRight, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotRightVector)
		// if err != nil {
		// 	panic(err)
		// }

		// fmt.Printf("i-th:%d, rotLeft:%d, rotRight:%d\n", i, rotNumber, (rotNumber-cipherTensorCols+Slots)%Slots)
		ctLeft, err := evaluator.RotateNew(cipherTensor.Ciphertexts[i], rotNumber)
		if err != nil {
			panic(err)
		}
		ctRight, err := evaluator.RotateNew(cipherTensor.Ciphertexts[i], (rotNumber-cipherTensorCols+Slots)%Slots)
		if err != nil {
			panic(err)
		}
		newCiphertextsLeft[i] = ctLeft
		newCiphertextsRight[i] = ctRight
	}

	// 返回结果
	return &encryption.CiphertextTensor{
			Ciphertexts: newCiphertextsLeft,
			NumRows:     cipherTensorRows,
			NumCols:     cipherTensorCols,
			NumDepth:    cipherTensorDepth,
		}, &encryption.CiphertextTensor{
			Ciphertexts: newCiphertextsRight,
			NumRows:     cipherTensorRows,
			NumCols:     cipherTensorCols,
			NumDepth:    cipherTensorDepth,
		}, nil
}

/*
 * CiphertextTensorRotationByColsNotAddMultiThread
 * Input:  PublicParametersKeys,CipherTensorLeft, CipherTensorRight CiphertextTensor,rot int, PlainMat [][]float64
 * Output: CipherTensorLeft, CipherTensorRight, error
 * Compute:ctTensor(a,b,c)  ----   [A,B,C,D,E,|0,0,0]
 *         CiphertextTensorLeft    [C,D,E,0,0,|0,A,B]
 *         CiphertextTensorRight   [0,0,0,A,B,|C,D,E]
 */
func CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(param *hefloat.Parameters, evaluator *hefloat.Evaluator, cipherTensorLeft, cipherTensorRight *encryption.CiphertextTensor, rotNumber int, Slots int) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherTensorRows := cipherTensorLeft.NumRows
	cipherTensorCols := cipherTensorLeft.NumCols
	cipherTensorDepth := cipherTensorLeft.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensorRows, cipherTensorCols, cipherTensorDepth)

	rotNumber = (rotNumber + cipherTensorCols) % cipherTensorCols
	// 判断条件，如果rot大于cols，则说明旋转超出一行
	if rotNumber > cipherTensorCols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the rot size %d is too large, require <%d", rotNumber, cipherTensorCols)
	}

	// fmt.Println(rotLeftVector)
	// fmt.Println(rotRightVector)
	// 进行计算
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensorDepth)
	for i := 0; i < cipherTensorDepth; i++ {
		ct := hefloat.NewCiphertext(*param, cipherTensorLeft.Ciphertexts[0].Degree(), cipherTensorLeft.Ciphertexts[0].Level())

		baseSize := 1.0
		rotLeftVector, rotRightVector := GeneratePlainVecLeftAndRight(cipherTensorRows, cipherTensorCols, rotNumber, baseSize)

		ctLeft, err := evaluator.MulRelinNew(cipherTensorLeft.Ciphertexts[i], rotLeftVector)
		if err != nil {
			panic(err)
		}
		ctRight, err := evaluator.MulRelinNew(cipherTensorRight.Ciphertexts[i], rotRightVector)
		if err != nil {
			panic(err)
		}

		res, err := evaluator.AddNew(ctLeft, ctRight)
		if err != nil {
			panic(err)
		}
		if err = evaluator.Rescale(res, res); err != nil {
			panic(err)
		}
		if err = evaluator.Add(ct, res, ct); err != nil {
			panic(err)
		}

		newCiphertexts[i] = ct
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
 * CiphertextTensorRotationByColsNotAddMultiThread
 * Input:  PublicParametersKeys,CipherTensorLeft, CipherTensorRight CiphertextTensor,rot int, PlainMat [][]float64
 * Output: CipherTensorLeft, CipherTensorRight, error
 * Compute:ctTensor(a,b,c)  ----   [A,B,C,D,E,|0,0,0]
 *         CiphertextTensorLeft    [C,D,E,0,0,|0,A,B]
 *         CiphertextTensorRight   [0,0,0,A,B,|C,D,E]
 */
func CipherTensorMulPlainMatWithLeftAndRightTensorMultiThread(param *hefloat.Parameters, evaluator *hefloat.Evaluator, cipherTensorLeft, cipherTensorRight *encryption.CiphertextTensor, PlainMat [][]float64, rotNumber int, Slots int) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherTensorRows := cipherTensorLeft.NumRows
	cipherTensorCols := cipherTensorLeft.NumCols
	cipherTensorDepth := cipherTensorLeft.NumDepth
	// fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensorRows, cipherTensorCols, cipherTensorDepth)

	rotNumber = (rotNumber + cipherTensorCols) % cipherTensorCols
	// fmt.Println(rotNumber)
	// 判断条件，如果rot大于cols，则说明旋转超出一行
	if rotNumber > cipherTensorCols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the rot size %d is too large, require <%d", rotNumber, cipherTensorCols)
	}

	// 判断条件，明文矩阵是一个方阵且Depth==rows
	if cipherTensorDepth != len(PlainMat) {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the Plain Matrix %d is not equal cipherTensor Depth %d", cipherTensorDepth, len(PlainMat))
	}

	// fmt.Println(rotLeftVector)
	// fmt.Println(rotRightVector)
	// 进行计算
	newCiphertexts := make([]*rlwe.Ciphertext, cipherTensorDepth)
	for i := 0; i < cipherTensorDepth; i++ {
		ct := hefloat.NewCiphertext(*param, cipherTensorLeft.Ciphertexts[0].Degree(), cipherTensorLeft.Ciphertexts[0].Level())
		for j := 0; j < cipherTensorDepth; j++ {
			rotLeftVector, rotRightVector := GeneratePlainVecLeftAndRight(cipherTensorRows, cipherTensorCols, rotNumber, PlainMat[j][i])
			ctLeft, err := evaluator.MulRelinNew(cipherTensorLeft.Ciphertexts[j], rotLeftVector)
			if err != nil {
				panic(err)
			}
			ctRight, err := evaluator.MulRelinNew(cipherTensorRight.Ciphertexts[j], rotRightVector)
			if err != nil {
				panic(err)
			}
			// fmt.Println("Alread Mul")
			ctLeft.Scale = ctRight.Scale
			res, err := evaluator.AddNew(ctLeft, ctRight)
			if err != nil {
				panic(err)
			}
			if err = evaluator.Rescale(res, res); err != nil {
				panic(err)
			}
			// fmt.Println("Alread Add")
			res.Scale = ct.Scale
			if err = evaluator.Add(ct, res, ct); err != nil {
				panic(err)
			}
		}
		newCiphertexts[i] = ct
	}

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensorRows,
		NumCols:     cipherTensorCols,
		NumDepth:    cipherTensorDepth,
	}, nil
}

func GeneratePlainVecLeftAndRight(cipherTensorRows, cipherTensorCols, rotNumber int, baseSize float64) ([]float64, []float64) {
	// 先生成明文向量
	rotLeftVector := make([]float64, cipherTensorRows*cipherTensorCols)
	rotRightVector := make([]float64, cipherTensorRows*cipherTensorCols)
	for i := 0; i < cipherTensorRows; i++ {
		for j := 0; j < cipherTensorCols; j++ {
			if j < (cipherTensorCols - rotNumber) {
				rotLeftVector[i*cipherTensorCols+j] = baseSize
				rotRightVector[i*cipherTensorCols+j] = 0
			} else {
				rotLeftVector[i*cipherTensorCols+j] = 0
				rotRightVector[i*cipherTensorCols+j] = baseSize
			}
		}
	}
	return rotLeftVector, rotRightVector
}

func GeneratePlainVecLeftAndRightWithVec(cipherTensorRows, cipherTensorCols, rotNumber int, baseVector []float64, baseSize float64) ([]float64, []float64, error) {

	// fmt.Printf("cipherTensorRows:%d, cipherTensorCols:%d, rotNumber:%d, length baseVector:%d\n", cipherTensorRows, cipherTensorCols, rotNumber, len(baseVector))
	// 先生成明文向量
	rotLeftVector := make([]float64, cipherTensorRows*cipherTensorCols)
	rotRightVector := make([]float64, cipherTensorRows*cipherTensorCols)
	for i := 0; i < cipherTensorRows; i++ {
		for j := 0; j < cipherTensorCols; j++ {
			if j < rotNumber {
				rotLeftVector[i*cipherTensorCols+j] = baseSize * baseVector[j]
				rotRightVector[i*cipherTensorCols+j] = 0
			} else {
				rotLeftVector[i*cipherTensorCols+j] = 0
				rotRightVector[i*cipherTensorCols+j] = baseSize * baseVector[j]
			}
		}
	}
	return rotLeftVector, rotRightVector, nil
}

// rotateMatrixColumns rotates each column of the matrix by rotNumber positions.
func rotateMatrixColumns(Mat [][]float64, rotNumber int) [][]float64 {
	if len(Mat) == 0 || len(Mat[0]) == 0 {
		return Mat // Return empty matrix if input is empty
	}

	rows := len(Mat)
	cols := len(Mat[0])
	rotNumber = (rotNumber + rows) % rows
	rotMat := make([][]float64, rows)
	for i := range rotMat {
		rotMat[i] = make([]float64, cols)
	}

	// Rotate each column
	for j := 0; j < cols; j++ {
		column := make([]float64, rows)
		for i := 0; i < rows; i++ {
			column[i] = Mat[i][j]
		}

		// Rotate the column
		n := len(column)
		rotNumber = rotNumber % n // Ensure rotNumber is within bounds
		if rotNumber != 0 {
			rotated := make([]float64, n)
			for i := 0; i < n; i++ {
				rotated[i] = column[(i+rotNumber)%n]
			}
			column = rotated
		}

		for i := 0; i < rows; i++ {
			rotMat[i][j] = column[i]
		}
	}

	return rotMat
}

// rotateMatrixColumns rotates each column of the matrix by rotNumber positions.
func rotateMatrixRows(Mat [][]float64, rotNumber int) [][]float64 {
	if len(Mat) == 0 || len(Mat[0]) == 0 {
		return Mat // Return empty matrix if input is empty
	}

	rows := len(Mat)
	cols := len(Mat[0])
	rotNumber = (rotNumber + cols) % cols
	rotMat := make([][]float64, rows)
	for i := range rotMat {
		rotMat[i] = make([]float64, cols)
	}

	// Rotate each column
	for j := 0; j < rows; j++ {
		Rows := make([]float64, cols)
		for i := 0; i < cols; i++ {
			Rows[i] = Mat[j][i]
		}

		// Rotate the column
		n := len(Rows)
		rotNumber = rotNumber % n // Ensure rotNumber is within bounds
		if rotNumber != 0 {
			rotated := make([]float64, n)
			for i := 0; i < n; i++ {
				rotated[i] = Rows[(i+rotNumber)%n]
			}
			Rows = rotated
		}

		for i := 0; i < cols; i++ {
			rotMat[j][i] = Rows[i]
		}
	}

	return rotMat
}

/*
 * CiphertextTensorQKVToAttentionWithBSGSMultiThread
 * Input:  PublicParametersKeys,Q,K,V CiphertextTensor
 * Output: *rlwe.Ciphertext,error
 * Compute: Q,K,V --> Attention result
 * 1CMul+1Mul
 */
func PlainMatMultiplyCiphertextTensorThenAdd(param *hefloat.Parameters, evaluator *hefloat.Evaluator, plainMat [][]float64, cipherTensor2 *encryption.CiphertextTensor) (*rlwe.Ciphertext, error) {

	// 确定进行密文矩阵×明文矩阵的维数
	plainRows := len(plainMat)
	plainCols := len(plainMat[0])
	// 判断QKV是否一样
	if plainCols != cipherTensor2.NumDepth {
		return &rlwe.Ciphertext{}, fmt.Errorf("NumCols mismatch: plainCols=%d, cipherTensor2.NumDepth=%d", plainCols, cipherTensor2.NumDepth)
	}

	ct := hefloat.NewCiphertext(*param, cipherTensor2.Ciphertexts[0].Degree(), cipherTensor2.Ciphertexts[0].Level())
	// 进行BSGS To Attetion
	for i := 0; i < cipherTensor2.NumDepth; i++ {
		plainVec := make([]float64, plainRows)
		for j := 0; j < plainRows; j++ {
			plainVec[j] = plainMat[j][i]
		}
		plainVec, err := utils.ReaptVector(plainVec, cipherTensor2.NumRows)
		if err != nil {
			panic(err)
		}

		err = evaluator.MulRelinThenAdd(cipherTensor2.Ciphertexts[i], plainVec, ct)
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
 * Output: *rlwe.Ciphertext,error
 * Compute: Q,K,V --> Attention result
 * 1CMul+1Mul
 */
func CiphertextTensorMultiplyPlainMatThenAdd(param *hefloat.Parameters, evaluator *hefloat.Evaluator, cipherTensor1 *encryption.CiphertextTensor, plainMat [][]float64) (*rlwe.Ciphertext, error) {

	// 确定进行密文矩阵×明文矩阵的维数
	plainRows := len(plainMat)
	plainCols := len(plainMat[0])
	// 判断QKV是否一样
	if plainRows != cipherTensor1.NumDepth {
		return &rlwe.Ciphertext{}, fmt.Errorf("NumCols mismatch: plainCols=%d, cipherTensor2.NumDepth=%d", plainRows, cipherTensor1.NumDepth)
	}

	ct := hefloat.NewCiphertext(*param, cipherTensor1.Ciphertexts[0].Degree(), cipherTensor1.Ciphertexts[0].Level())
	// 进行BSGS To Attetion
	for i := 0; i < cipherTensor1.NumDepth; i++ {
		plainVec := make([]float64, plainCols)
		for j := 0; j < plainCols; j++ {
			plainVec[j] = plainMat[i][j]
		}
		plainVec, err := utils.ReaptVector(plainVec, cipherTensor1.NumRows)
		if err != nil {
			panic(err)
		}

		err = evaluator.MulRelinThenAdd(cipherTensor1.Ciphertexts[i], plainVec, ct)
		if err != nil {
			panic(err)
		}

	}
	evaluator.Rescale(ct, ct)
	// 返回结果
	return ct, nil
}

func GetDiagRotVector(mat [][]float64, i_th int, rotNumber int) ([]float64, error) {
	matRows := len(mat)
	matCols := len(mat[0])
	if matRows != matCols {
		return []float64{}, fmt.Errorf("matrix is not a square matrix (%d,%d)", matRows, matCols)
	}
	resVec := make([]float64, matRows)
	rotNumber = (rotNumber + matRows) % matRows
	for i := 0; i < matRows; i++ {
		resVec[i] = mat[(i+rotNumber)%matRows][(i_th+i+rotNumber)%matCols]
	}
	return resVec, nil
}

func GenerateCipherTensorRot(publicKeys *encryption.PublicParametersKeys, X0 *encryption.CiphertextTensor, babyStep, giantStep int) ([]*encryption.CiphertextTensor, []*encryption.CiphertextTensor, []*encryption.CiphertextTensor, []*encryption.CiphertextTensor, error) {
	// 声明并初始化用于存储旋转结果的数组
	var X0RotTensorLeft = make([]*encryption.CiphertextTensor, giantStep)
	var X0RotTensorRight = make([]*encryption.CiphertextTensor, giantStep)
	var X0RotTensor = make([]*encryption.CiphertextTensor, giantStep)
	var X0TRotTensor = make([]*encryption.CiphertextTensor, babyStep)

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex

	// 生成旋转所有的步长
	for i := 0; i < giantStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 旋转Q
			// fmt.Printf("Q:%d,K:%d,V:%d\n", -i*babyStep, i, i)
			rotX0Left, rotX0Right, err := CipherTensorRotationByColsNotAddMultiThread(evaluator, X0, -i*babyStep, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			rotX0, err := CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, rotX0Left, rotX0Right, -i*babyStep, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			if i < babyStep {
				rotX0T, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, X0, i, 1, publicKeys.Params.MaxSlots())
				if err != nil {
					panic(err)
				}
				X0TRotTensor[i] = rotX0T
			}
			// mu.Lock()
			X0RotTensorLeft[i] = rotX0Left
			X0RotTensorRight[i] = rotX0Right
			X0RotTensor[i] = rotX0
			// mu.Unlock()
		}(i)
	}
	wg.Wait()

	return X0RotTensor, X0TRotTensor, X0RotTensorLeft, X0RotTensorRight, nil
}

func GenerateCipherTensorRot_bak(publicKeys *encryption.PublicParametersKeys, X0 *encryption.CiphertextTensor, babyStep, giantStep int) ([]*encryption.CiphertextTensor, []*encryption.CiphertextTensor, []*encryption.CiphertextTensor, []*encryption.CiphertextTensor, error) {
	// 声明并初始化用于存储旋转结果的数组
	var X0RotTensorLeft = make([]*encryption.CiphertextTensor, giantStep)
	var X0RotTensorRight = make([]*encryption.CiphertextTensor, giantStep)
	var X0RotTensor = make([]*encryption.CiphertextTensor, giantStep)
	var X0TRotTensor = make([]*encryption.CiphertextTensor, babyStep)

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex

	// 生成旋转所有的步长
	for i := 0; i < giantStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 旋转Q
			// fmt.Printf("Q:%d,K:%d,V:%d\n", -i*babyStep, i, i)
			rotX0Left, rotX0Right, err := CipherTensorRotationByColsNotAddMultiThread(evaluator, X0, -i*babyStep, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			rotX0, err := CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, rotX0Left, rotX0Right, -i*babyStep, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}

			rotX0T, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, X0, i, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			// mu.Lock()
			X0RotTensorLeft[i] = rotX0Left
			X0RotTensorRight[i] = rotX0Right
			X0RotTensor[i] = rotX0
			if i < babyStep {
				X0TRotTensor[i] = rotX0T
			}
			// mu.Unlock()
		}(i)
	}
	wg.Wait()
	return X0RotTensor, X0TRotTensor, X0RotTensorLeft, X0RotTensorRight, nil
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
func PlainVecCipherTensorMulPlainMatWithRotationByColsNewMultiThread(param *hefloat.Parameters, evaluator *hefloat.Evaluator, cipherTensor *encryption.CiphertextTensor, baseVector []float64, plainMat [][]float64, rotNumber int, Slots int) (*encryption.CiphertextTensor, error) {

	// 返回密文张量维数
	cipherTensorRows := cipherTensor.NumRows
	cipherTensorCols := cipherTensor.NumCols
	cipherTensorDepth := cipherTensor.NumDepth
	fmt.Printf("Ciphertext Tensor Rows:%d, Cols:%d, Depths:%d\n", cipherTensorRows, cipherTensorCols, cipherTensorDepth)

	// 确定进行密文矩阵×明文矩阵的维数
	plainRows := len(plainMat)
	plainCols := len(plainMat[0])
	fmt.Printf("Plain Mat Rows:%d, Cols:%d\n", plainRows, plainCols)
	rotNumber = (rotNumber + cipherTensorCols) % cipherTensorCols
	fmt.Printf("Rot Number:%d\n", rotNumber)
	// 判断条件，如果rot大于cols，则说明旋转超出一行
	if rotNumber >= cipherTensorCols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the rot size %d is too large, require <%d", rotNumber, cipherTensorCols)
	}

	if len(baseVector) != cipherTensorCols || plainRows != cipherTensorDepth {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the baseVector length %d is not equal cipherTensorCols %d", len(baseVector), cipherTensorCols)
	}

	newCiphertextsLeft := make([]*rlwe.Ciphertext, cipherTensorDepth)
	newCiphertextsRight := make([]*rlwe.Ciphertext, cipherTensorDepth)
	for i := 0; i < cipherTensorDepth; i++ {
		// ctLeft, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotLeftVector)
		// if err != nil {
		// 	panic(err)
		// }
		// ctRight, err := evaluator.MulRelinNew(cipherTensor.Ciphertexts[i], rotRightVector)
		// if err != nil {
		// 	panic(err)
		// }

		// fmt.Printf("i-th:%d, rotLeft:%d, rotRight:%d\n", i, rotNumber, (rotNumber-cipherTensorCols+Slots)%Slots)
		ctLeft, err := evaluator.RotateNew(cipherTensor.Ciphertexts[i], rotNumber)
		if err != nil {
			panic(err)
		}
		ctRight, err := evaluator.RotateNew(cipherTensor.Ciphertexts[i], (rotNumber-cipherTensorCols+Slots)%Slots)
		if err != nil {
			panic(err)
		}
		newCiphertextsLeft[i] = ctLeft
		newCiphertextsRight[i] = ctRight
	}

	// 先生成明文向量
	newCiphertexts := make([]*rlwe.Ciphertext, plainCols)
	for i := 0; i < plainCols; i++ {
		ct := hefloat.NewCiphertext(*param, newCiphertextsLeft[0].Degree(), newCiphertextsLeft[0].Level())
		for j := 0; j < cipherTensorDepth; j++ {
			rotLeftVector, rotRightVector, err := GeneratePlainVecLeftAndRightWithVec(cipherTensorRows, cipherTensorCols, rotNumber, baseVector, plainMat[j][i])
			if err != nil {
				panic(err)
			}
			ctLeft, err := evaluator.MulRelinNew(newCiphertextsLeft[j], rotLeftVector)
			if err != nil {
				panic(err)
			}
			ctRight, err := evaluator.MulRelinNew(newCiphertextsRight[j], rotRightVector)
			if err != nil {
				panic(err)
			}
			// fmt.Println("Alread Mul")
			ctLeft.Scale = ctRight.Scale
			res, err := evaluator.AddNew(ctLeft, ctRight)
			if err != nil {
				panic(err)
			}
			if err = evaluator.Rescale(res, res); err != nil {
				panic(err)
			}
			// fmt.Println("Alread Add")
			res.Scale = ct.Scale
			if err = evaluator.Add(ct, res, ct); err != nil {
				panic(err)
			}
		}
		newCiphertexts[i] = ct
	}

	// fmt.Println(rotLeftVector)
	// fmt.Println(rotRightVector)
	// 进行计算

	// 返回结果
	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensorRows,
		NumCols:     cipherTensorCols,
		NumDepth:    plainCols,
	}, nil
}

func CiphertextTensorAddPlaintextMatrixWithEvaluator(evaluator *hefloat.Evaluator, ciphertextTensor *encryption.CiphertextTensor, plainSlice [][]float64) (*encryption.CiphertextTensor, error) {
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
		newCiphertexts[i], err = evaluator.AddNew(ciphertextTensor.Ciphertexts[i], plainVector)
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

package maths

import (
	"dashformer/encryption"
	"dashformer/utils"
	"fmt"
	"runtime"
	"sync"
	"testing"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func CiphertextTensorQKVToAttentionWithBSGSMultiThreadCopy(publicKeys *encryption.PublicParametersKeys, Q, K, V *encryption.CiphertextTensor, babyStep, giantStep int, b, c float64) (*encryption.CiphertextTensor, error) {

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
			// rotK, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, K, i, 1/(math.Sqrt(32)*math.Sqrt(c)), publicKeys.Params.MaxSlots())
			rotK, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, K, i, 1, publicKeys.Params.MaxSlots())
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
					// diagMatrixSoftMax, err := ApproximateSoftmaxCiphertext(evaluator, diagMatrix, b/math.Sqrt(c), 1)
					// if err != nil {
					// 	panic(err)
					// }
					err = CiphertextTensorMultiplyCiphertextTensorAddToRes(publicKeys.Params, evaluator, diagMatrix, VRotTensor[j], QKV)
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
func TestBSGS(t *testing.T) {
	// 初始化参数和密钥对（假设已经定义好）

	publicKeys, secretKeys, err := encryption.SetHERealParams()
	// publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		panic(err)
	}

	plainTensorValue := [][][]float64{
		{
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
			{1.7, 1.8, 1.9},
			{2.1, 2.2, 2.3},
			{1.1, 1.2, 1.3},
			{1.4, 1.5, 1.6},
		},
	}

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

	fmt.Printf("===========================\n")
	fmt.Printf("Testing Rot\n")
	fmt.Printf("===========================\n")
	fmt.Printf("\n")
	// 测试BSGS功能
	newCiphertextTensor2, err := CiphertextTensorRotationByColsNewMultiThread(publicKeys.Evaluator, ciphertextTensor, -1, 1, publicKeys.Params.MaxSlots())
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor1 Level:%d\n", newCiphertextTensor2.Ciphertexts[0].Level())
	valueTensor, err := encryption.DecryptTensorValue(secretKeys, newCiphertextTensor2)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensor, "Rot")
	fmt.Println(valueTensor)

	fmt.Printf("===========================\n")
	fmt.Printf("Testing Attention with BSGS\n")
	fmt.Printf("===========================\n")
	fmt.Printf("\n")
	// 测试BSGS功能
	newCiphertextTensor1, err := CiphertextTensorQKVToAttentionWithBSGSMultiThreadCopy(publicKeys, ciphertextTensor, ciphertextTensor, ciphertextTensor, 7, 8, 0.96, 200)
	if err != nil {
		panic(err)
	}

	// 解析密文
	fmt.Printf("Ciphertext Tensor1 Level:%d\n", newCiphertextTensor1.Ciphertexts[0].Level())
	valueTensor, err = encryption.DecryptTensorValue(secretKeys, newCiphertextTensor1)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensor, "Attention")
	fmt.Println(valueTensor[0])

}

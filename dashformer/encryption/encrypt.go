package encryption

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type CiphertextTensor struct {
	Ciphertexts []*rlwe.Ciphertext
	NumRows     int
	NumCols     int
	NumDepth    int
}

// ShallowCopy 方法为 CiphertextTensor 结构体实现浅拷贝
func (ct *CiphertextTensor) ShallowCopy() *CiphertextTensor {
	// 创建新对象并复制基本字段
	newCiphertextTensor := &CiphertextTensor{
		Ciphertexts: make([]*rlwe.Ciphertext, len(ct.Ciphertexts)),
		NumRows:     ct.NumRows,
		NumCols:     ct.NumCols,
		NumDepth:    ct.NumDepth,
	}

	// 逐个拷贝 Ciphertexts 切片中的每个 Ciphertext
	for idx, ciphertext := range ct.Ciphertexts {
		if ciphertext != nil {
			newCiphertextTensor.Ciphertexts[idx] = ciphertext.CopyNew()
		}
	}

	return newCiphertextTensor
}

/*
 * EncryptTensorValue
 * Input:  PublicParametersKeys,ptTensor Slice[][][]
 * Output: CiphertextTensor,error
 * Compute: Encrypting ptTensor to ctTensor
 */
func EncryptTensorValue(publicKeys *PublicParametersKeys, plainTensorValue [][][]float64) (*CiphertextTensor, error) {
	numRows := len(plainTensorValue)
	numCols := len(plainTensorValue[0])
	numDepth := len(plainTensorValue[0][0])

	ciphertexts := make([]*rlwe.Ciphertext, numDepth)

	for d := 0; d < numDepth; d++ {
		// 提取第三维的一个切片
		plainSlice := make([]float64, numRows*numCols)
		index := 0
		for i := 0; i < numRows; i++ {
			for j := 0; j < numCols; j++ {
				plainSlice[index] = plainTensorValue[i][j][d]
				index++
			}
		}
		// fmt.Println(plainSlice)

		// 加密该切片
		pt := hefloat.NewPlaintext(*publicKeys.Params, publicKeys.Params.MaxLevel())
		if err := publicKeys.Encoder.Encode(plainSlice, pt); err != nil {
			panic(err)
		}
		ct, err := publicKeys.Encryptor.EncryptNew(pt)
		if err != nil {
			panic(err)
		}
		ciphertexts[d] = ct
	}

	return &CiphertextTensor{
		Ciphertexts: ciphertexts,
		NumRows:     numRows,
		NumCols:     numCols,
		NumDepth:    numDepth,
	}, nil
}

/*
 * EncryptTensorValueMultiTread
 * Input:  PublicParametersKeys,ptTensor Slice[][][]
 * Output: CiphertextTensor,error
 * Compute: Encrypting ptTensor to ctTensor
 */
func EncryptTensorValueMultiTread(publicKeys *PublicParametersKeys, plainTensorValue [][][]float64) (*CiphertextTensor, error) {
	numRows := len(plainTensorValue)
	numCols := len(plainTensorValue[0])
	numDepth := len(plainTensorValue[0][0])

	ciphertexts := make([]*rlwe.Ciphertext, numDepth)

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex

	for d := 0; d < numDepth; d++ {
		wg.Add(1)
		// 提取第三维的一个切片
		go func(d int) {
			defer wg.Done()
			encoder := publicKeys.Encoder.ShallowCopy()
			encryptor := publicKeys.Encryptor.ShallowCopy()
			plainSlice := make([]float64, numRows*numCols)
			index := 0
			for i := 0; i < numRows; i++ {
				for j := 0; j < numCols; j++ {
					plainSlice[index] = plainTensorValue[i][j][d]
					index++
				}
			}
			// fmt.Println(plainSlice)

			// 加密该切片
			pt := hefloat.NewPlaintext(*publicKeys.Params, publicKeys.Params.MaxLevel())
			if err := encoder.Encode(plainSlice, pt); err != nil {
				panic(err)
			}
			ct, err := encryptor.EncryptNew(pt)
			if err != nil {
				panic(err)
			}
			// mu.Lock()
			ciphertexts[d] = ct
			// mu.Unlock()
		}(d)
	}
	wg.Wait()

	return &CiphertextTensor{
		Ciphertexts: ciphertexts,
		NumRows:     numRows,
		NumCols:     numCols,
		NumDepth:    numDepth,
	}, nil
}

/*
 * MergeAndAddCiphertextTensors
 * Input: tensor1 *CiphertextTensor , tensor2 *CiphertextTensor
 * Output: *CiphertextTensor,error
 * Compute: tensor1(a,b,c)  tensor2(a,b,d) --> tensor(a,b,c+d)
 */
func MergeAndAddCiphertextTensors(tensor1, tensor2 *CiphertextTensor) (*CiphertextTensor, error) {
	// 检查是否有一个 tensor 为空
	if tensor1 == nil || len(tensor1.Ciphertexts) == 0 {
		return tensor2, nil
	}
	if tensor2 == nil || len(tensor2.Ciphertexts) == 0 {
		return tensor1, nil
	}

	if tensor1.NumRows != tensor2.NumRows || tensor1.NumCols != tensor2.NumCols {
		return &CiphertextTensor{}, fmt.Errorf("CipherTensor can not add, tensor1 is (%d,%d), tensor2 is (%d,%d)", tensor1.NumRows, tensor1.NumCols, tensor2.NumRows, tensor2.NumCols)
	}
	// 合并 Ciphertexts 列表
	mergedCiphertexts := append(tensor1.Ciphertexts, tensor2.Ciphertexts...)

	// 相加 NumDepth
	mergedNumDepth := tensor1.NumDepth + tensor2.NumDepth

	return &CiphertextTensor{
		Ciphertexts: mergedCiphertexts,
		NumRows:     tensor1.NumRows, // 根据需要选择适当的 NumRows 和 NumCols
		NumCols:     tensor1.NumCols, // 根据需要选择适当的 NumCols
		NumDepth:    mergedNumDepth,
	}, nil
}

/*
 * AddTwoCipherTensorNew
 * Input:  tensor1 *CiphertextTensor , tensor2 *CiphertextTensor
 * Output: *CiphertextTensor,error
 * Compute: Merge and Add ciphertextTensor
 */
func AddTwoCipherTensorNew(publicKeys *PublicParametersKeys, tensor1, tensor2 *CiphertextTensor) (*CiphertextTensor, error) {
	// 检查是否有一个 tensor 为空
	if tensor1 == nil || len(tensor1.Ciphertexts) == 0 {
		return tensor2, nil
	}
	if tensor2 == nil || len(tensor2.Ciphertexts) == 0 {
		return tensor1, nil
	}

	if tensor1.NumRows != tensor2.NumRows || tensor1.NumCols != tensor2.NumCols || tensor1.NumDepth != tensor2.NumDepth {
		return &CiphertextTensor{}, fmt.Errorf("CipherTensor can not add, tensor1 is (%d,%d,%d), tensor2 is (%d,%d,%d)", tensor1.NumRows, tensor1.NumCols, tensor1.NumDepth, tensor2.NumRows, tensor2.NumCols, tensor2.NumDepth)
	}
	// 对密文进行相加
	newCiphertexts := make([]*rlwe.Ciphertext, tensor1.NumDepth)
	var err error
	for i := 0; i < tensor1.NumDepth; i++ {
		// tensor2.Ciphertexts[i].Scale = tensor1.Ciphertexts[i].Scale
		newCiphertexts[i], err = publicKeys.Evaluator.AddNew(tensor1.Ciphertexts[i], tensor2.Ciphertexts[i])
		if err != nil {
			panic(err)
		}
	}

	return &CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     tensor1.NumRows, // 根据需要选择适当的 NumRows 和 NumCols
		NumCols:     tensor1.NumCols, // 根据需要选择适当的 NumCols
		NumDepth:    tensor1.NumDepth,
	}, nil
}

/*
 * AddTwoCipherTensorNewMultiThread
 * Input:  tensor1 *CiphertextTensor , tensor2 *CiphertextTensor
 * Output: *CiphertextTensor,error
 * Compute: Merge and Add ciphertextTensor
 */
func AddTwoCipherTensorNewMultiThread(publicKeys *PublicParametersKeys, tensor1, tensor2 *CiphertextTensor) (*CiphertextTensor, error) {
	// 检查是否有一个 tensor 为空
	if tensor1 == nil || len(tensor1.Ciphertexts) == 0 {
		return tensor2, nil
	}
	if tensor2 == nil || len(tensor2.Ciphertexts) == 0 {
		return tensor1, nil
	}

	if tensor1.NumRows != tensor2.NumRows || tensor1.NumCols != tensor2.NumCols || tensor1.NumDepth != tensor2.NumDepth {
		return &CiphertextTensor{}, fmt.Errorf("CipherTensor can not add, tensor1 is (%d,%d), tensor2 is (%d,%d)", tensor1.NumRows, tensor1.NumCols, tensor2.NumRows, tensor2.NumCols)
	}
	// 对密文进行相加
	newCiphertexts := make([]*rlwe.Ciphertext, tensor1.NumDepth)
	var err error
	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// // 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex
	for i := 0; i < tensor1.NumDepth; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// tensor2.Ciphertexts[i].Scale = tensor1.Ciphertexts[i].Scale
			newCiphertexts[i], err = evaluator.AddNew(tensor1.Ciphertexts[i], tensor2.Ciphertexts[i])
			if err != nil {
				panic(err)
			}
		}(i)
	}
	wg.Wait()

	return &CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     tensor1.NumRows, // 根据需要选择适当的 NumRows 和 NumCols
		NumCols:     tensor1.NumCols, // 根据需要选择适当的 NumCols
		NumDepth:    tensor1.NumDepth,
	}, nil
}

/*
 * AddThreeCipherTensorNewMultiThread
 * Input:  tensor1 *CiphertextTensor, tensor2 *CiphertextTensor, tensor3 *CiphertextTensor
 * Output: *CiphertextTensor, error
 * Compute: Merge and Add three CiphertextTensors
 */
func AddThreeCipherTensorNewMultiThread(publicKeys *PublicParametersKeys, tensor1, tensor2, tensor3 *CiphertextTensor) (*CiphertextTensor, error) {
	// 检查是否有一个 tensor 为空
	if tensor1 == nil || len(tensor1.Ciphertexts) == 0 {
		return AddTwoCipherTensorNewMultiThread(publicKeys, tensor2, tensor3)
	}
	if tensor2 == nil || len(tensor2.Ciphertexts) == 0 {
		return AddTwoCipherTensorNewMultiThread(publicKeys, tensor1, tensor3)
	}
	if tensor3 == nil || len(tensor3.Ciphertexts) == 0 {
		return AddTwoCipherTensorNewMultiThread(publicKeys, tensor1, tensor2)
	}

	// 检查张量维度是否一致
	if tensor1.NumRows != tensor2.NumRows || tensor1.NumCols != tensor2.NumCols || tensor1.NumDepth != tensor2.NumDepth ||
		tensor1.NumRows != tensor3.NumRows || tensor1.NumCols != tensor3.NumCols || tensor1.NumDepth != tensor3.NumDepth {
		return &CiphertextTensor{}, fmt.Errorf("CipherTensor dimensions do not match: tensor1 is (%d,%d,%d), tensor2 is (%d,%d,%d), tensor3 is (%d,%d,%d)",
			tensor1.NumRows, tensor1.NumCols, tensor1.NumDepth,
			tensor2.NumRows, tensor2.NumCols, tensor2.NumDepth,
			tensor3.NumRows, tensor3.NumCols, tensor3.NumDepth)
	}

	// 对密文进行相加
	newCiphertexts := make([]*rlwe.Ciphertext, tensor1.NumDepth)
	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup

	for i := 0; i < tensor1.NumDepth; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			// 先将 tensor1 和 tensor2 相加
			sum, err := evaluator.AddNew(tensor1.Ciphertexts[i], tensor2.Ciphertexts[i])
			if err != nil {
				panic(err)
			}
			// 再将上一步的结果与 tensor3 相加
			newCiphertexts[i], err = evaluator.AddNew(sum, tensor3.Ciphertexts[i])
			if err != nil {
				panic(err)
			}
		}(i)
	}
	wg.Wait()

	return &CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     tensor1.NumRows,
		NumCols:     tensor1.NumCols,
		NumDepth:    tensor1.NumDepth,
	}, nil
}

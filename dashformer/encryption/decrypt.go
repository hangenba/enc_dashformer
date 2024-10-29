package encryption

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func DecryptTensorValue(secretKeys *SecretParametersKeys, ciphertextTensor *CiphertextTensor) ([][][]float64, error) {

	// 初始化三维张量valueTensor
	valueTensor := make([][][]float64, ciphertextTensor.NumRows)
	for i := range valueTensor {
		valueTensor[i] = make([][]float64, ciphertextTensor.NumCols)
		for j := range valueTensor[i] {
			valueTensor[i][j] = make([]float64, ciphertextTensor.NumDepth)
		}
	}

	for i, ct := range ciphertextTensor.Ciphertexts {
		pt := secretKeys.Decryptor.DecryptNew(ct)
		// Decodes the plaintext
		have := make([]float64, secretKeys.Params.MaxSlots())
		if err := secretKeys.Encoder.Decode(pt, have); err != nil {
			panic(err)
		}
		for j := 0; j < ciphertextTensor.NumRows; j++ {
			for k := 0; k < ciphertextTensor.NumCols; k++ {
				valueTensor[j][k][i] = have[j*ciphertextTensor.NumCols+k]
			}
		}
	}
	return valueTensor, nil
}

func DecryptTensorValueMultiThread(secretKeys *SecretParametersKeys, ciphertextTensor *CiphertextTensor) ([][][]float64, error) {

	// 初始化三维张量valueTensor
	valueTensor := make([][][]float64, ciphertextTensor.NumRows)
	for i := range valueTensor {
		valueTensor[i] = make([][]float64, ciphertextTensor.NumCols)
		for j := range valueTensor[i] {
			valueTensor[i][j] = make([]float64, ciphertextTensor.NumDepth)
		}
	}

	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex

	for i, ct := range ciphertextTensor.Ciphertexts {
		wg.Add(1)
		go func(i int, ct *rlwe.Ciphertext) {
			defer wg.Done()
			decryptor := secretKeys.Decryptor.ShallowCopy()
			encoder := secretKeys.Encoder.ShallowCopy()

			pt := decryptor.DecryptNew(ct)
			// Decodes the plaintext
			have := make([]float64, secretKeys.Params.MaxSlots())
			if err := encoder.Decode(pt, have); err != nil {
				panic(err)
			}
			// mu.Lock()
			// defer mu.Unlock()
			for j := 0; j < ciphertextTensor.NumRows; j++ {
				for k := 0; k < ciphertextTensor.NumCols; k++ {
					valueTensor[j][k][i] = have[j*ciphertextTensor.NumCols+k]
				}
			}
		}(i, ct)
	}
	wg.Wait()
	return valueTensor, nil
}

func DecryptVectorValue(secretKeys *SecretParametersKeys, ct *rlwe.Ciphertext) {

	var err error

	// Decrypts the vector of plaintext values
	pt := secretKeys.Decryptor.DecryptNew(ct)

	// Decodes the plaintext
	have := make([]float64, secretKeys.Params.MaxSlots())
	if err = secretKeys.Encoder.Decode(pt, have); err != nil {
		panic(err)
	}

	// Pretty prints some values
	fmt.Printf("Have: ")
	for i := 0; i < 10; i++ {
		fmt.Printf("%20.15f ", have[i])
	}
	fmt.Printf("...\n")

}

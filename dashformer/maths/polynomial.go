package maths

import (
	"dashformer/encryption"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/utils/bignum"
)

// 目前出现问题：在计算方差和均值的过程中，scale会扩大，不进行rescale会出现问题，但是进行rescale层数又会增加
func ApproximatePolynomial(publicKeys *encryption.PublicParametersKeys, ct *rlwe.Ciphertext, coeffs []float64, domain [2]float64) (*rlwe.Ciphertext, error) {
	poly := bignum.NewPolynomial(bignum.Basis(0), coeffs, domain)
	polyEval := hefloat.NewPolynomialEvaluator(*publicKeys.Params, publicKeys.Evaluator)
	// fmt.Println(publicKeys.Params.LogDefaultScale())
	// fmt.Println(ct.LogScale())

	res, err := polyEval.Evaluate(ct, poly, ct.Scale)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func ApproximatePolynomialCipherTensor(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, coeffs []float64, domain [2]float64) (*encryption.CiphertextTensor, error) {
	poly := bignum.NewPolynomial(bignum.Basis(0), coeffs, domain)
	polyEval := hefloat.NewPolynomialEvaluator(*publicKeys.Params, publicKeys.Evaluator)
	// fmt.Printf("poly: %v\n", poly)

	// fmt.Println(ciphertextTensor.Ciphertexts[0].LogScale())
	ciphertexts := make([]*rlwe.Ciphertext, ciphertextTensor.NumDepth)
	for i, ct := range ciphertextTensor.Ciphertexts {
		res, err := polyEval.Evaluate(ct, poly, ct.Scale)
		if err != nil {
			return nil, err
		}
		ciphertexts[i] = res
	}
	return &encryption.CiphertextTensor{
		Ciphertexts: ciphertexts,
		NumRows:     ciphertextTensor.NumRows,
		NumCols:     ciphertextTensor.NumCols,
		NumDepth:    ciphertextTensor.NumDepth,
	}, nil
}

func ApproximatePolynomialCipherTensorMultiThread(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, coeffs []float64, domain [2]float64) (*encryption.CiphertextTensor, error) {
	// fmt.Println(publicKeys.Params.DefaultScale())
	// fmt.Println(ciphertextTensor.Ciphertexts[0].Scale)
	// fmt.Println(ciphertextTensor.Ciphertexts[0].Degree())
	// 设置最大使用的操作系统线程数为 8
	runtime.GOMAXPROCS(4)
	// 创建一个 WaitGroup 来等待所有 goroutine 完成
	var wg sync.WaitGroup
	// 使用一个互斥锁来保护对 newCiphertexts 的并发访问
	// var mu sync.Mutex
	// fmt.Println(ciphertextTensor.Ciphertexts[0].LogScale())
	ciphertexts := make([]*rlwe.Ciphertext, ciphertextTensor.NumDepth)
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			poly := bignum.NewPolynomial(bignum.Basis(0), coeffs, domain)
			polyEval := hefloat.NewPolynomialEvaluator(*publicKeys.Params, evaluator)

			res, err := polyEval.Evaluate(ciphertextTensor.Ciphertexts[i], poly, publicKeys.Params.DefaultScale())
			if err != nil {
				panic(err)
			}

			// mu.Lock()
			ciphertexts[i] = res
			// mu.Unlock()
		}(i)
	}

	wg.Wait()
	return &encryption.CiphertextTensor{
		Ciphertexts: ciphertexts,
		NumRows:     ciphertextTensor.NumRows,
		NumCols:     ciphertextTensor.NumCols,
		NumDepth:    ciphertextTensor.NumDepth,
	}, nil
}

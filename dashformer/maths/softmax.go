package maths

import (
	"dashformer/encryption"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func ApproximateSoftmax(publicKeys *encryption.PublicParametersKeys, ciphertextTensor *encryption.CiphertextTensor, b float64, c float64) (*encryption.CiphertextTensor, error) {
	// Softmax激活函数逻辑
	// softmax = (x+b)**2/c
	newCiphertexts := make([]*rlwe.Ciphertext, ciphertextTensor.NumDepth)
	for i := 0; i < ciphertextTensor.NumDepth; i++ {
		ctAdd, err := publicKeys.Evaluator.AddNew(ciphertextTensor.Ciphertexts[i], b)
		if err != nil {
			panic(err)
		}
		ctSqrt, err := publicKeys.Evaluator.MulRelinNew(ctAdd, ctAdd)
		if err != nil {
			panic(err)
		}
		publicKeys.Evaluator.Rescale(ctSqrt, ctSqrt)

		// ctres, err := publicKeys.Evaluator.MulRelinNew(ctSqrt, float64(1/float64(400)))
		// if err != nil {
		// 	panic(err)
		// }
		// publicKeys.Evaluator.Rescale(ctres, ctres)

		// 这里将除以常数，整合到前面的式子中进行运算，将乘1/c-->在1/sqrt(32)位置同时进行
		newCiphertexts[i] = ctSqrt
	}

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     ciphertextTensor.NumRows,
		NumCols:     ciphertextTensor.NumCols,
		NumDepth:    ciphertextTensor.NumDepth,
	}, nil
}

func ApproximateSoftmaxCiphertext(evaluator *hefloat.Evaluator, ct *rlwe.Ciphertext, b float64, c float64) (*rlwe.Ciphertext, error) {
	// Softmax激活函数逻辑
	// softmax = (x+b)**2/c
	ctAdd, err := evaluator.AddNew(ct, b)
	if err != nil {
		panic(err)
	}
	ctSqrt, err := evaluator.MulRelinNew(ctAdd, ctAdd)
	if err != nil {
		panic(err)
	}
	evaluator.Rescale(ctSqrt, ctSqrt)

	return ctSqrt, nil
}

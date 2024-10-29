package maths

import (
	"dashformer/encryption"
	"dashformer/utils"
	"fmt"
	"testing"
)

func TestReLU(t *testing.T) {
	// 初始化参数和密钥对

	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		panic(err)
	}

	plainTensorValue := [][][]float64{
		{
			{-1558.499141465721, 0, -3.3},
			{4.4, 8, -0.3},
		},
		{
			{1, 28, 3},
			{-45, 4, 0.12},
		},
	}
	//加密
	ciphertextTensor, err := encryption.EncryptTensorValue(publicKeys, plainTensorValue)
	if err != nil {
		panic(err)
	}
	if len(ciphertextTensor.Ciphertexts) != len(plainTensorValue[0][0]) {
		t.Errorf("Expected %d ciphertexts, got %d", len(plainTensorValue[0][0]), len(ciphertextTensor.Ciphertexts))
	}
	fmt.Printf("Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)

	fmt.Printf("Ciphertext Tensor Degree:%d\n", ciphertextTensor.Ciphertexts[0].Level())
	fmt.Printf("\n\nTesting ciphertext tensor ReLU ......\n")

	// reluCoefficients := []float64{
	// 	6.37605427e-01, // x^0
	// 	3.43799506e-01, // x^1
	// 	5.61612124e-02, // x^2
	// 	3.84343820e-03, // x^3
	// 	1.15689566e-04, // x^4
	// 	1.26115207e-06, // x^5
	// }
	// reluCoefficients := []float64{
	// 	6.80680382e-01,  // x^0
	// 	3.83068933e-01,  // x^1
	// 	5.89317628e-02,  // x^2
	// 	2.79235443e-03,  // x^3
	// 	-5.31682993e-05, // x^4
	// 	-8.21002091e-06, // x^5
	// 	-2.30147919e-07, // x^6
	// 	-2.04664795e-09, // x^7
	// }

	reluCoefficients := []float64{
		9.43637250e-01,
		3.57849530e-01,
		3.64713370e-02,
		1.12135100e-03,
		-7.30229137e-06,
		-7.32668212e-07,
		-7.02786376e-09,
	}
	//进行密文ReLU
	newCiphertextReLU_1, err := ApproximatePolynomialCipherTensorMultiThread(publicKeys, ciphertextTensor, reluCoefficients, [2]float64{0, 1})
	if err != nil {
		panic(err)
	}
	//解析密文
	fmt.Printf("Ciphertext Tensor1 Level:%d\n", newCiphertextReLU_1.Ciphertexts[0].Level())
	decryptedTensor, err := encryption.DecryptTensorValue(secretKeys, newCiphertextReLU_1)
	if err != nil {
		panic(err)
	}

	utils.PrintSliceInfo(decryptedTensor, "decryptedTensor")
	fmt.Println(decryptedTensor)
}

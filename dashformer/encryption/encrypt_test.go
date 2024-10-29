package encryption

import (
	"dashformer/utils"
	"fmt"
	"testing"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"golang.org/x/exp/rand"
)

func TestEncryptAndDecryptTensorValue(t *testing.T) {
	// 初始化参数和密钥对（假设已经定义好）

	publicKeys, secretKeys, err := SetHERealParams()
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

	ciphertextTensor, err := EncryptTensorValueMultiTread(publicKeys, plainTensorValue)
	if err != nil {
		panic(err)
	}
	if len(ciphertextTensor.Ciphertexts) != len(plainTensorValue[0][0]) {
		t.Errorf("Expected %d ciphertexts, got %d", len(plainTensorValue[0][0]), len(ciphertextTensor.Ciphertexts))
	}
	fmt.Printf("Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)

	// 测试旋转操作
	fmt.Printf("================\n")
	fmt.Printf("Testing Rotation\n")
	fmt.Printf("================\n")
	fmt.Printf("\n")

	// Vector of plaintext values
	Slots := publicKeys.Params.MaxSlots()
	values := make([]float64, Slots)
	want := make([]float64, Slots)
	r := rand.New(rand.NewSource(0))
	// Populates the vector of plaintext values
	for i := range values {
		values[i] = 2*r.Float64() - 1 // uniform in [-1, 1]
	}
	// Encodes the vector of plaintext values
	pt := hefloat.NewPlaintext(*publicKeys.Params, publicKeys.Params.MaxLevel())
	if err = publicKeys.Encoder.Encode(values, pt); err != nil {
		panic(err)
	}
	// Encrypts the vector of plaintext values
	ctTest, err := publicKeys.Encryptor.EncryptNew(pt)
	if err != nil {
		panic(err)
	}
	var rot = (-3 + Slots) % Slots
	// Rotation by 5 positions to the left
	for i := 0; i < Slots; i++ {
		want[i] = values[(i+rot)%Slots]
	}
	ctRot, err := publicKeys.Evaluator.RotateNew(ctTest, rot)
	if err != nil {
		panic(err)
	}
	pt = secretKeys.Decryptor.DecryptNew(ctRot)
	have := make([]float64, publicKeys.Params.MaxSlots())
	if err = secretKeys.Encoder.Decode(pt, have); err != nil {
		panic(err)
	}
	fmt.Printf("Rotation by k=%d %s", rot, hefloat.GetPrecisionStats(*publicKeys.Params, publicKeys.Encoder, secretKeys.Decryptor, want, ctRot, 0, false).String())
	fmt.Println(values[:10])
	fmt.Println(want[:10])
	fmt.Println(have[:10])

	// 解析密文
	valueTensor, err := DecryptTensorValueMultiThread(secretKeys, ciphertextTensor)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensor, "valueTensor")
	fmt.Println(valueTensor)

	// 测试相加操作
	fmt.Printf("======================================\n")
	fmt.Printf("Testing Add Two Ciphertext MultiThread\n")
	fmt.Printf("======================================\n")
	fmt.Printf("\n")

	addCipherTensor, err := AddTwoCipherTensorNewMultiThread(publicKeys, ciphertextTensor, ciphertextTensor)
	if err != nil {
		panic(err)
	}
	// 解析密文
	valueTensor, err = DecryptTensorValueMultiThread(secretKeys, addCipherTensor)
	if err != nil {
		panic(err)
	}
	utils.PrintSliceInfo(valueTensor, "valueTensor")
	fmt.Println(valueTensor)
}

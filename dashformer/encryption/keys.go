package encryption

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type PublicParametersKeys struct {
	Params    *hefloat.Parameters
	Encoder   *hefloat.Encoder
	Encryptor *rlwe.Encryptor
	Evaluator *hefloat.Evaluator
}

type SecretParametersKeys struct {
	Params    *hefloat.Parameters
	Sk        *rlwe.SecretKey
	Encoder   *hefloat.Encoder
	Decryptor *rlwe.Decryptor
}

func CopyKeyGenerator(params rlwe.ParameterProvider, enc *rlwe.Encryptor) *rlwe.KeyGenerator {
	return &rlwe.KeyGenerator{
		Encryptor: enc,
	}
}

func SetHERealParams() (*PublicParametersKeys, *SecretParametersKeys, error) {
	fmt.Printf("CKKS initialization ...")
	ckksIniStartTime := time.Now()
	var err error
	var params hefloat.Parameters
	if params, err = hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN: 14,
			LogQ: []int{38, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33},
			LogP: []int{36, 36},
			// RingType:        ring.ConjugateInvariant,
			LogDefaultScale: 33,
		}); err != nil {
		panic(err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	ecd := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)

	// 生成旋转步数
	var rotNumbers []int
	for i := -50; i <= 50; i++ {
		rotNumbers = append(rotNumbers, i)
	}
	galEls := params.GaloisElements(rotNumbers)
	// fmt.Printf("galEls: %v\n", galEls)
	// eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

	// // BEGIN

	// single thread
	// galoisKeys := make([]*rlwe.GaloisKey, len(galEls))
	// for i, galEl := range galEls {
	// 	if galoisKeys[i] == nil {
	// 		galoisKeys[i] = kgen.GenGaloisKeyNew(galEl, sk)
	// 	} else {
	// 		kgen.GenGaloisKey(galEl, sk, galoisKeys[i])
	// 	}
	// }
	// fmt.Printf("Galois keys generation ... completed\n")

	// FOUR threads

	galoisKeys := make([]*rlwe.GaloisKey, len(galEls))
	runtime.GOMAXPROCS(4)
	var wg sync.WaitGroup
	numThreads := 4
	chunkSize := (len(galEls) + numThreads - 1) / numThreads
	// fmt.Printf("dimension of galEls: %d\n", len(galEls))
	// fmt.Printf("dimension of chunkSize: %d\n", chunkSize)

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > len(galEls) {
			end = len(galEls)
		}
		// fmt.Printf("(start, end): (%d, %d)\n", start, end)
		wg.Add(1)
		go func(start, end, threadNum int) {
			defer wg.Done()
			localEncryptor := enc.ShallowCopy()
			localKgen := CopyKeyGenerator(params, localEncryptor)

			// localKgen := OurKeyGenerator(params, enc)
			for i := start; i < end; i++ {
				galEl := galEls[i]
				// fmt.Printf("galEl @ thread %d: %d\n", threadNum, galEl)
				if galoisKeys[i] == nil {
					// fmt.Printf("I AM thread %d ... OK\n    -- current i: %d\n", threadNum, i)
					galoisKeys[i] = localKgen.GenGaloisKeyNew(galEl, sk)
				} else {
					// fmt.Printf("I AM thread %d ... OK\n", threadNum)
					kgen.GenGaloisKey(galEl, sk, galoisKeys[i])
				}
				// fmt.Printf("Generated Galois Key for rotation %d\n", galEl)
			}
		}(start, end, t)
	}

	wg.Wait()

	eval := hefloat.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...))
	// // fmt.Println("I AM HERE")
	// // END

	dec := rlwe.NewDecryptor(params, sk)
	fmt.Printf(" takes %s\n", time.Since(ckksIniStartTime))
	fmt.Printf("  - log N = %d, log Q = %d, max_level = %d, log_scale = %d\n",
		params.LogN(), int(params.LogQP()), params.MaxLevel(), params.LogDefaultScale())

	return &PublicParametersKeys{
			Params:    &params,
			Encoder:   ecd,
			Encryptor: enc,
			Evaluator: eval,
		}, &SecretParametersKeys{
			Params:    &params,
			Sk:        sk,
			Encoder:   ecd,
			Decryptor: dec,
		}, nil
}

// func SetBSGSHERealParams(babyStep int, giantStep int, cols int) (*PublicParametersKeys, *SecretParametersKeys, error) {
// 	var err error
// 	var params hefloat.Parameters

// 	if params, err = hefloat.NewParametersFromLiteral(
// 		hefloat.ParametersLiteral{
// 			LogN:            13,
// 			LogQ:            []int{51, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
// 			LogP:            []int{50, 50, 50},
// 			LogDefaultScale: 40,
// 		}); err != nil {
// 		panic(err)
// 	}

// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKeyNew()
// 	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
// 	rlk := kgen.GenRelinearizationKeyNew(sk)
// 	evk := rlwe.NewMemEvaluationKeySet(rlk)
// 	ecd := hefloat.NewEncoder(params)
// 	enc := rlwe.NewEncryptor(params, pk)
// 	eval := hefloat.NewEvaluator(params, evk)

// 	// 生成旋转步数
// 	var rotNumbers []int
// 	// 生成大步小步所需的步长
// 	for i := 1; i*babyStep < cols; i++ {
// 		rotNumbers = append(rotNumbers, i)
// 		rotNumbers = append(rotNumbers, -i)
// 		rotNumbers = append(rotNumbers, i*babyStep)
// 		rotNumbers = append(rotNumbers, -i*babyStep)
// 	}
// 	fmt.Print(rotNumbers)
// 	galEls := params.GaloisElements(rotNumbers)
// 	fmt.Println(galEls)
// 	// 假如pooling旋转密钥
// 	// galEls = append(galEls, params.GaloisElementsForInnerSum(1, cols)...)
// 	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

// 	dec := rlwe.NewDecryptor(params, sk)

// 	return &PublicParametersKeys{
// 			Params:    &params,
// 			Encoder:   ecd,
// 			Encryptor: enc,
// 			Evaluator: eval,
// 		}, &SecretParametersKeys{
// 			Params:    &params,
// 			Sk:        sk,
// 			Encoder:   ecd,
// 			Decryptor: dec,
// 		}, nil
// }

package config

import (
	"fmt"
)

// Init initializes the configuration paths and returns them.
func Init() (string, string, string, string) {
	fmt.Println("Initializing configuration...")

	// 初始化文件读取地址
	examplePath := "data/example_AA_sequences.list"
	tokenizerPath := "data/dashformer_tokenizer.json"
	modelParamPath := "data/dashformer_model_parameters"
	outputPath := "data/output"

	return examplePath, tokenizerPath, modelParamPath, outputPath
}

func InitCoeffients() ([]float64, []float64, []float64, [4]float64, [4]float64) {
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
	// reluCoefficients := []float64{
	// 	9.30593456e-01,  // x^0
	// 	3.48227788e-01,  // x^1
	// 	3.60751072e-02,  // x^2
	// 	1.21317400e-03,  // x^3
	// 	-2.43517943e-06, // x^4
	// 	-8.01397704e-07, // x^5
	// 	-1.27534065e-08, // x^6
	// 	-5.80639394e-11, // x^7
	// }
	// reluCoefficients := []float64{
	// 	9.43637250e-01,
	// 	3.57849530e-01,
	// 	3.64713370e-02,
	// 	1.12135100e-03,
	// 	-7.30229137e-06,
	// 	-7.32668212e-07,
	// 	-7.02786376e-09,
	// }
	sqrtLayerCoefficients1 := []float64{
		4.01447285e-01,
		-1.41122823e-02,
		3.37694161e-04,
		-4.54776425e-06,
		3.15551268e-08,
		-8.73491970e-11,
	}
	sqrtLayerCoefficients2 := []float64{
		4.62876515e-01,
		-1.77386329e-02,
		3.71793457e-04,
		-3.71871638e-06,
		1.69876334e-08,
		-2.83686695e-11,
	}

	// softMaxB := [4]float64{0.95, 1, 0.8, 1.3}
	// softMaxC := [4]float64{478, 226, 161, 375}
	// softMaxB := [4]float64{1.95, 1.95, 1.95, 1.95}
	// softMaxC := [4]float64{200, 200, 200, 200}

	softMaxB := [4]float64{1.32, 0.75, 0.66, 1.14}
	softMaxC := [4]float64{450, 181, 158, 376}
	reluCoefficients := []float64{
		9.43651501e-01,
		3.59049720e-01,
		3.66350473e-02,
		1.12737776e-03,
		-7.22653539e-06,
		-7.31025115e-07,
		-6.99022399e-09,
	}

	return reluCoefficients, sqrtLayerCoefficients1, sqrtLayerCoefficients2, softMaxB, softMaxC
}

package coefficient

import (
	"fmt"
	"reflect"
)

type Coefficient_input struct {
	One_50_row    []float64
	One_50_coloum [][]float64

	W_e [][]float64
	P   [][]float64

	W_Q [4][][]float64
	B_Q [4][]float64
	W_K [4][][]float64
	B_K [4][]float64
	W_V [4][][]float64
	B_V [4][]float64

	W_c [][]float64 //
	B_c []float64   //

	W_1 [][]float64
	B_1 []float64
	W_2 [][]float64
	B_2 []float64

	Sigma_1_diag []float64
	Sigma_2_diag []float64
	Sigma_1      [][]float64 // 50
	Sigma_2      [][]float64
	Gamma_1      [][]float64 // 128 X 128
	Gamma_2      [][]float64
	Beta_1       []float64
	Beta_2       []float64

	W_d [][]float64 // 128 X 25
	B_d []float64
}

type Coefficient_dash struct {
	Relu_before []float64
	Relu_rear   [][]float64

	Head_before_relu []float64
	Head_rear_relu   [][]float64

	Head_before []float64
	Head_rear   [][]float64

	X0_before []float64
	X0_rear   [][]float64

	X0_before_relu []float64
	X0_rear_relu   [][]float64

	Constant_Dash []float64
	Constant_Relu [][]float64
}

// type Coefficient_head struct {

// 	Sigma_1_diag []float64

// 	W_Head [][][]float64 // len = 4
// }

type Coefficient_QKV struct {
	A_Q [][][]float64 // 4 X 25 X 32
	A_K [][][]float64
	A_V [][][]float64

	Constant_Q [][][]float64 // 4 X 50 X 32
	Constant_K [][][]float64
	Constant_V [][][]float64
}

type Coefficient_sqmax struct {

	// G_g []float64 // len = 4

	Item_1 [][][]float64 // 4, include g
	Item_2 [][][]float64 // 4, include g
	Item_3 [][][]float64 // 4, include g
	Item_4 [][][]float64 // 4, constant_sqmax
}

// 获取字段的维数
func getDimensions(value reflect.Value) []int {
	var dimensions []int

	switch value.Kind() {
	case reflect.Slice, reflect.Array:
		dimensions = append(dimensions, value.Len())
		if value.Len() > 0 && (value.Index(0).Kind() == reflect.Slice || value.Index(0).Kind() == reflect.Array) {
			dimensions = append(dimensions, getDimensions(value.Index(0))...)
		}
	}
	return dimensions
}

// 打印结构体所有字段的维数
func (d *Coefficient_input) PrintDimensions() {
	v := reflect.ValueOf(*d)
	typeOfS := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		dimensions := getDimensions(field)

		fmt.Printf("%s: %v\n", typeOfS.Field(i).Name, dimensions)
	}
}

func (d *Coefficient_dash) PrintDimensions() {
	v := reflect.ValueOf(*d)
	typeOfS := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		dimensions := getDimensions(field)

		fmt.Printf("%s: %v\n", typeOfS.Field(i).Name, dimensions)
	}
}

// func (d *Coefficient_head) PrintDimensions() {
// 	v := reflect.ValueOf(*d)
// 	typeOfS := v.Type()

// 	for i := 0; i < v.NumField(); i++ {
// 		field := v.Field(i)
// 		dimensions := getDimensions(field)

// 		fmt.Printf("%s: %v\n", typeOfS.Field(i).Name, dimensions)
// 	}
// }

func (d *Coefficient_sqmax) PrintDimensions() {
	v := reflect.ValueOf(*d)
	typeOfS := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		dimensions := getDimensions(field)

		fmt.Printf("%s: %v\n", typeOfS.Field(i).Name, dimensions)
	}
}

func (d *Coefficient_QKV) PrintDimensions() {
	v := reflect.ValueOf(*d)
	typeOfS := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		dimensions := getDimensions(field)

		fmt.Printf("%s: %v\n", typeOfS.Field(i).Name, dimensions)
	}
}

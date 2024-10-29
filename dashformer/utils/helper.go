package utils

import (
	"fmt"
	"reflect"
)

// PrintSliceInfo 函数接受一个二维切片，直接在函数内部输出切片的名字和维数
func PrintSliceInfo(slice interface{}, name string) {
	// 使用反射获取切片的值
	v := reflect.ValueOf(slice)
	if v.Kind() != reflect.Slice {
		fmt.Printf("错误: %s 不是一个切片\n", name)
		return
	}

	// 获取切片的维数
	dims := []int{}
	for v.Kind() == reflect.Slice {
		dims = append(dims, v.Len())
		if v.Len() > 0 {
			v = v.Index(0)
		} else {
			break
		}
	}

	// // 输出第一个元素的值
	// if len(dims) > 0 && dims[0] > 0 {
	// 	fmt.Printf("%s[0]: %v\n", name, reflect.ValueOf(slice).Index(0))
	// 	fmt.Println(reflect.ValueOf(slice).Index(0).Len())
	// }
	// 输出结果
	fmt.Printf("%s: %v\n", name, dims)
}

// ReapteVector 函数接受向量和重复次数，输出len(vector)*numRepeats长度的向量
func ReaptVector(baseVector []float64, numRepeat int) ([]float64, error) {
	extendedVector := make([]float64, 0, numRepeat*len(baseVector))
	for i := 0; i < numRepeat; i++ {
		extendedVector = append(extendedVector, baseVector...)
	}
	return extendedVector, nil
}

// 定义一个通用的函数，对向量中的每个元素乘以一个值
func ScaleVector(vector []float64, multiplier float64) []float64 {
	newVector := make([]float64, len(vector))
	for i := range vector {
		newVector[i] = vector[i] * multiplier
	}
	return newVector
}

// 定义一个通用的函数，对向量中的每个元素乘以一个值
func ScaleMatrix(mat [][]float64, multiplier float64) [][]float64 {
	//newMat := make([][]float64, len(mat))
	for i := range len(mat) {
		for j := range len(mat[0]) {
			mat[i][j] = mat[i][j] * multiplier
		}
	}
	return mat
}

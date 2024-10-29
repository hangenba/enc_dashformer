package coefficient

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Compute_Gamma(d float64, diag []float64) ([][]float64) {
	length := len(diag)
	
	result := CreateIdentityMatrix(length)
	result.Scale(d, result)
	result.Sub(result, CreateOnesMatrix(length,length))
	result.Mul(result, CreateDiagonalMatrix(diag))
	return DenseToSlice(result)
}

func MatrixChainMultiply_matdense(matrices []*mat.Dense) (*mat.Dense, error) {
	if len(matrices) == 0 {
		return nil, fmt.Errorf("no matrices provided")
	}

	// 初始化 result
	result := mat.DenseCopyOf(matrices[0])

	for i := 1; i < len(matrices); i++ {
		var temp mat.Dense
		temp.Mul(result, matrices[i])
		result = mat.DenseCopyOf(&temp) // 深拷贝到result
	}

	return result, nil
}

func MatrixChainMultiply_slice(data ...interface{}) ([][]float64) {
	
	var matrices []*mat.Dense
	for i:=0;i<len(data);i++ {
		ma, _ := ConvertSliceToMatDense(data[i])
		matrices = append(matrices, ma)
	}
	result, _ := MatrixChainMultiply_matdense(matrices)

	return DenseToSlice(result)
}

func addMatrices(a, b [][]float64) ([][]float64, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("matrices have different number of rows")
	}

	result := make([][]float64, len(a))
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return nil, fmt.Errorf("matrices have different number of columns in row %d", i)
		}
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}

	return result, nil
}

func MatrixChainAdd_slice(data ...[][]float64) ([][]float64) {
    
	if len(data) == 1 {
		return data[0]
	}
    result, _ := addMatrices(data[0], data[1])
    for i:=2;i<len(data);i++ {
        result, _ = addMatrices(result, data[i])
    }
    return result
}


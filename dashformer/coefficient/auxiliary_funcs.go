package coefficient

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func DenseToSlice(dense *mat.Dense) [][]float64 {
    rows, cols := dense.Dims()

    slice := make([][]float64, rows)
    for i := range slice {
        slice[i] = make([]float64, cols)
    }

    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            slice[i][j] = dense.At(i, j)
        }
    }
    return slice
}

func CreateOnesMatrix(rows, cols int) *mat.Dense {
    data := make([]float64, rows*cols)

    for i := range data {
        data[i] = 1.0
    }

    return mat.NewDense(rows, cols, data)
}

func CreateIdentityMatrix(size int) *mat.Dense {
    identity := mat.NewDense(size, size, nil)

    for i := 0; i < size; i++ {
        identity.Set(i, i, 1.0)
    }
    return identity
}

func CreateDiagonalMatrix(diag []float64) *mat.Dense {
    size := len(diag)
    
    diagonalMatrix := mat.NewDense(size, size, nil)

    for i := 0; i < size; i++ {
        diagonalMatrix.Set(i, i, diag[i])
    }
    return diagonalMatrix
}

func ReadSigma(filename string) ([]float64, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("无法打开文件: %v", err)
    }
    defer file.Close()

    var data []float64

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := scanner.Text()
        value, err := strconv.ParseFloat(line, 64)
        if err != nil {
            return nil, fmt.Errorf("转换错误: %v", err)
        }
        data = append(data, value)
    }

    if err := scanner.Err(); err != nil {
        return nil, fmt.Errorf("读取文件时发生错误: %v", err)
    }
    return data, nil
}

func ToDiagonalMatrix(vec []float64) [][]float64 {
    n := len(vec)
    matrix := make([][]float64, n)
    for i := range matrix {
        matrix[i] = make([]float64, n)
    }

    for i := 0; i < n; i++ {
        matrix[i][i] = vec[i]
    }
    return matrix
}

func ConvertSliceToMatDense(data interface{}) (*mat.Dense, error) {
	switch v := data.(type) {
	case []float64:
		n := len(v)
		return mat.NewDense(1, n, v), nil
	case [][]float64:
		rows := len(v)
		if rows == 0 {
			return nil, fmt.Errorf("empty 2D slice")
		}
		cols := len(v[0])
		if cols == 0 {
			return nil, fmt.Errorf("2D slice has empty rows")
		}

		flattened := make([]float64, 0, rows*cols)
		for _, row := range v {
			if len(row) != cols {
				return nil, fmt.Errorf("inconsistent row lengths")
			}
			flattened = append(flattened, row...)
		}
		return mat.NewDense(rows, cols, flattened), nil
	default:
		return nil, fmt.Errorf("unsupported data type")
	}
}

func Transp(matrix [][]float64) [][]float64 {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return [][]float64{}
    }
    
    rows := len(matrix)
    cols := len(matrix[0])
    
    transposed := make([][]float64, cols)
    for i := range transposed {
        transposed[i] = make([]float64, rows)
    }

    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            transposed[j][i] = matrix[i][j]
        }
    }
    
    return transposed
}

func MultiplyByScalar(matrix [][]float64, scalar float64) [][]float64 {
    rows := len(matrix)
    if rows == 0 {
        return [][]float64{}
    }
    cols := len(matrix[0])
    
    result := make([][]float64, rows)
    for i := range result {
        result[i] = make([]float64, cols)
    }
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            result[i][j] = matrix[i][j] * scalar
        }
    }
    
    return result
}

func AddWithScalar(matrix [][]float64, scalar float64) [][]float64 {
    rows := len(matrix)
    if rows == 0 {
        return [][]float64{}
    }
    cols := len(matrix[0])
    
    result := make([][]float64, rows)
    for i := range result {
        result[i] = make([]float64, cols)
    }
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            result[i][j] = matrix[i][j] + scalar
        }
    }
    
    return result
}

func SplitMatrixIntoFourChunks_byRow(matrix [][]float64) [][][]float64 {
    rows := len(matrix)
    if rows == 0 {
        return [][][]float64{}
    }

    if rows%4 != 0 {
        fmt.Println("Matrix cannot be evenly split into 4 chunks!")
        return [][][]float64{}
    }

    chunkSize := rows / 4
    
    result := make([][][]float64, 4)
    
    for i := 0; i < 4; i++ {
        start := i * chunkSize
        end := (i + 1) * chunkSize
        result[i] = matrix[start:end]
    }
    return result
}

func InverseMatrix(matrix [][]float64) ([][]float64, error) {

    rows := len(matrix)
    if rows == 0 || len(matrix[0]) != rows {
        return nil, fmt.Errorf("matrix is not square")
    }

    data := make([]float64, 0, rows*rows)
    for i := 0; i < rows; i++ {
        data = append(data, matrix[i]...)
    }
    
    m := mat.NewDense(rows, rows, data)
    
    var inverse mat.Dense
    err := inverse.Inverse(m)
    if err != nil {
        return nil, fmt.Errorf("matrix is singular or not invertible")
    }

    invData := make([][]float64, rows)
    for i := 0; i < rows; i++ {
        invData[i] = make([]float64, rows)
        for j := 0; j < rows; j++ {
            invData[i][j] = inverse.At(i, j)
        }
    }
    return invData, nil
}
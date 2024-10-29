package utils

import (
	"fmt"
	"reflect"
)

// 定义 DashformerModelParameters 结构体
type DashformerModelParameters struct {
	EmbeddingMatrix [][]float64
	EncodingMatrix  [][]float64

	QueryWeightAttentionMatrixs [4][][]float64
	QueryBiasAttentionVectors   [4][]float64
	KeyWeightAttentionMatrixs   [4][][]float64
	KeyBiasAttentionVectors     [4][]float64
	ValueWeightAttentionMatrixs [4][][]float64
	ValueBiasAttentionVectors   [4][]float64

	CombineWeightMatrixs [][]float64
	CombineBiasVectors   []float64

	LayerNormVectorR1 []float64
	LayerNormVectorB1 []float64
	LayerNormVectorR2 []float64
	LayerNormVectorB2 []float64

	FeedForwardWeightMatrix1 [][]float64
	FeedForwardBiasVector1   []float64
	FeedForwardWeightMatrix2 [][]float64
	FeedForwardBiasVector2   []float64

	ClassifierWeightMatrix [][]float64
	ClassifierBiasVector   []float64

	ReluCoefficients       []float64
	SqrtLayerCoefficients1 []float64
	SqrtLayerCoefficients2 []float64

	LayerNormSqrtVariance1 []float64
	LayerNormSqrtVariance2 []float64

	SoftMaxB [4]float64
	SoftMaxC [4]float64
}

// 生成 embeddingMatrix 的赋值函数
func (d *DashformerModelParameters) SetEmbeddingMatrix(value [][]float64) {
	d.EmbeddingMatrix = value
}

// 生成 encodingMatrix 的赋值函数
func (d *DashformerModelParameters) SetEncodingMatrix(value [][]float64) {
	d.EncodingMatrix = value
}

// 生成 queryWeightAttentionMatrixs 的赋值函数
func (d *DashformerModelParameters) SetQueryWeightAttentionMatrixs(value [4][][]float64) {
	d.QueryWeightAttentionMatrixs = value
}

// 生成 queryBiasAttentionVectors 的赋值函数
func (d *DashformerModelParameters) SetQueryBiasAttentionVectors(value [4][]float64) {
	d.QueryBiasAttentionVectors = value
}

// 生成 keyWeightAttentionMatrixs 的赋值函数
func (d *DashformerModelParameters) SetKeyWeightAttentionMatrixs(value [4][][]float64) {
	d.KeyWeightAttentionMatrixs = value
}

// 生成 keyBiasAttentionVectors 的赋值函数
func (d *DashformerModelParameters) SetKeyBiasAttentionVectors(value [4][]float64) {
	d.KeyBiasAttentionVectors = value
}

// 生成 valueWeightAttentionMatrixs 的赋值函数
func (d *DashformerModelParameters) SetValueWeightAttentionMatrixs(value [4][][]float64) {
	d.ValueWeightAttentionMatrixs = value
}

// 生成 valueBiasAttentionVectors 的赋值函数
func (d *DashformerModelParameters) SetValueBiasAttentionVectors(value [4][]float64) {
	d.ValueBiasAttentionVectors = value
}

// 生成 combineWeightMatrixs 的赋值函数
func (d *DashformerModelParameters) SetCombineWeightMatrixs(value [][]float64) {
	d.CombineWeightMatrixs = value
}

// 生成 combineBiasVectors 的赋值函数
func (d *DashformerModelParameters) SetCombineBiasVectors(value []float64) {
	d.CombineBiasVectors = value
}

// 生成 layerNormVectorR1 的赋值函数
func (d *DashformerModelParameters) SetLayerNormVectorR1(value []float64) {
	d.LayerNormVectorR1 = value
}

// 生成 LayerNormVectorB1 的赋值函数
func (d *DashformerModelParameters) SetLayerNormVectorB1(value []float64) {
	d.LayerNormVectorB1 = value
}

// 生成 layerNormVectorR2 的赋值函数
func (d *DashformerModelParameters) SetLayerNormVectorR2(value []float64) {
	d.LayerNormVectorR2 = value
}

// 生成 LayerNormVectorB2 的赋值函数
func (d *DashformerModelParameters) SetLayerNormVectorB2(value []float64) {
	d.LayerNormVectorB2 = value
}

// 生成 feedForwardWeightMatrix1 的赋值函数
func (d *DashformerModelParameters) SetFeedForwardWeightMatrix1(value [][]float64) {
	d.FeedForwardWeightMatrix1 = value
}

// 生成 feedForwardBiasVector1 的赋值函数
func (d *DashformerModelParameters) SetFeedForwardBiasVector1(value []float64) {
	d.FeedForwardBiasVector1 = value
}

// 生成 feedForwardWeightMatrix2 的赋值函数
func (d *DashformerModelParameters) SetFeedForwardWeightMatrix2(value [][]float64) {
	d.FeedForwardWeightMatrix2 = value
}

// 生成 feedForwardBiasVector2 的赋值函数
func (d *DashformerModelParameters) SetFeedForwardBiasVector2(value []float64) {
	d.FeedForwardBiasVector2 = value
}

// 生成 classifierWeightMatrix 的赋值函数
func (d *DashformerModelParameters) SetClassifierWeightMatrix(value [][]float64) {
	d.ClassifierWeightMatrix = value
}

// 生成 classifierBiasVector 的赋值函数
func (d *DashformerModelParameters) SetClassifierBiasVector(value []float64) {
	d.ClassifierBiasVector = value
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
func (d *DashformerModelParameters) PrintDimensions() {
	v := reflect.ValueOf(*d)
	typeOfS := v.Type()

	fmt.Printf("=================================\n")
	fmt.Printf("Print Dashformer Model Parameters\n")
	fmt.Printf("=================================\n")
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		dimensions := getDimensions(field)

		fmt.Printf("%s: %v\n", typeOfS.Field(i).Name, dimensions)
	}
}

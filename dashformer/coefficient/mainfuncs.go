package coefficient

import (
	"dashformer/utils"
	"fmt"
	"math"
)

func Create_Coefficient_input(dash utils.DashformerModelParameters) Coefficient_input {

	d := 128.0

	sigma_1_diag := dash.LayerNormSqrtVariance1
	sigma_2_diag := dash.LayerNormSqrtVariance2
	for i := 0; i < len(sigma_1_diag); i++ {
		sigma_1_diag[i] = sigma_1_diag[i] / d
	}
	for i := 0; i < len(sigma_2_diag); i++ {
		sigma_2_diag[i] = sigma_2_diag[i] / d
	}

	gam_1 := Compute_Gamma(d, dash.LayerNormVectorR1)
	gam_2 := Compute_Gamma(d, dash.LayerNormVectorR2)

	one_row := make([]float64, 50)
	for i := range one_row {
		one_row[i] = 1.0
	}
	one_coloum := make([][]float64, 50)
	for i := 0; i < 50; i++ {
		one_coloum[i] = append(one_coloum[i], 1.0)
	}

	dash.ClassifierWeightMatrix = utils.ScaleMatrix(dash.ClassifierWeightMatrix, 1/2649.372705)
	dash.ClassifierBiasVector = utils.ScaleVector(dash.ClassifierBiasVector, 1/2649.372705)

	return Coefficient_input{
		One_50_row:    one_row,
		One_50_coloum: one_coloum,

		W_e: dash.EmbeddingMatrix,
		P:   dash.EncodingMatrix,

		W_Q: dash.QueryWeightAttentionMatrixs,
		B_Q: dash.QueryBiasAttentionVectors,
		W_K: dash.KeyWeightAttentionMatrixs,
		B_K: dash.KeyBiasAttentionVectors,
		W_V: dash.ValueWeightAttentionMatrixs,
		B_V: dash.ValueBiasAttentionVectors,

		W_c: dash.CombineWeightMatrixs,
		B_c: dash.CombineBiasVectors,

		W_1: dash.FeedForwardWeightMatrix1,
		B_1: dash.FeedForwardBiasVector1,
		W_2: dash.FeedForwardWeightMatrix2,
		B_2: dash.FeedForwardBiasVector2,

		Sigma_1_diag: sigma_1_diag,
		Sigma_2_diag: sigma_2_diag,
		Sigma_1:      ToDiagonalMatrix(sigma_1_diag),
		Sigma_2:      ToDiagonalMatrix(sigma_2_diag),
		Gamma_1:      gam_1,
		Gamma_2:      gam_2,
		Beta_1:       dash.LayerNormVectorB1,
		Beta_2:       dash.LayerNormVectorB2,

		W_d: dash.ClassifierWeightMatrix,
		B_d: dash.ClassifierBiasVector,
	}
}

func Compute_coefficient_dash(in Coefficient_input) Coefficient_dash {

	c_y2 := MatrixChainAdd_slice(MatrixChainMultiply_slice(in.Sigma_1, in.One_50_coloum, in.B_c, in.Gamma_1),
		MatrixChainMultiply_slice(in.Sigma_1, in.P, in.Gamma_1),
		MatrixChainMultiply_slice(in.One_50_coloum, in.Beta_1))

	c_relu := MatrixChainAdd_slice(MatrixChainMultiply_slice(c_y2, in.W_1),
		MatrixChainMultiply_slice(in.One_50_coloum, in.B_1))

	c_dash := MatrixChainAdd_slice(MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, c_y2, in.Gamma_2, in.W_d),
		MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, in.One_50_coloum, in.B_2, in.Gamma_2, in.W_d),
		MatrixChainMultiply_slice(in.One_50_row, in.One_50_coloum, in.Beta_2, in.W_d))[0]
	for i := 0; i < 25; i++ {
		c_dash[i] = c_dash[i] + in.B_d[i]
	}

	// inverse, err := InverseMatrix(MatrixChainMultiply_slice(in.W_1, Transp(in.W_1)) )
	// if err != nil {
	// 	panic(err)
	// }

	return Coefficient_dash{
		Relu_before: MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2)[0],
		Relu_rear:   MatrixChainMultiply_slice(in.W_2, in.Gamma_2, in.W_d),

		Head_before_relu: in.Sigma_1_diag,
		Head_rear_relu:   MatrixChainMultiply_slice(in.W_c, in.Gamma_1, in.W_1),

		Head_before: MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, in.Sigma_1)[0],
		Head_rear:   MatrixChainMultiply_slice(in.W_c, in.Gamma_1, in.Gamma_2, in.W_d),
		X0_before:   MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, in.Sigma_1)[0],
		X0_rear:     MatrixChainMultiply_slice(in.W_e, in.Gamma_1, in.Gamma_2, in.W_d),

		X0_before_relu: in.Sigma_1_diag,
		X0_rear_relu:   MatrixChainMultiply_slice(in.W_e, in.Gamma_1, in.W_1),

		Constant_Dash: c_dash,
		Constant_Relu: c_relu,
	}
}

func Compute_coefficient_QKV(in Coefficient_input) Coefficient_QKV {

	var a_Q [][][]float64
	var a_K [][][]float64
	var a_V [][][]float64
	for i := 0; i < 4; i++ {
		a_Q = append(a_Q, MatrixChainMultiply_slice(in.W_e, in.W_Q[i]))
		a_K = append(a_K, MatrixChainMultiply_slice(in.W_e, in.W_K[i]))
		a_V = append(a_V, MatrixChainMultiply_slice(in.W_e, in.W_V[i]))
	}

	var constant_Q [][][]float64
	var constant_K [][][]float64
	var constant_V [][][]float64
	for i := 0; i < 4; i++ {
		constant_Q = append(constant_Q, MatrixChainAdd_slice(MatrixChainMultiply_slice(in.P, in.W_Q[i]), MatrixChainMultiply_slice(in.One_50_coloum, in.B_Q[i])))
		constant_K = append(constant_K, MatrixChainAdd_slice(MatrixChainMultiply_slice(in.P, in.W_K[i]), MatrixChainMultiply_slice(in.One_50_coloum, in.B_K[i])))
		constant_V = append(constant_V, MatrixChainAdd_slice(MatrixChainMultiply_slice(in.P, in.W_V[i]), MatrixChainMultiply_slice(in.One_50_coloum, in.B_V[i])))
	}

	return Coefficient_QKV{
		A_Q: a_Q,
		A_K: a_K,
		A_V: a_V,

		Constant_Q: constant_Q,
		Constant_K: constant_K,
		Constant_V: constant_V,
	}
}

func Compute_coefficient_sqmax(coeffi_QKV Coefficient_QKV, b [4]float64, c [4]float64) Coefficient_sqmax {

	if len(b) != 4 || len(c) != 4 {
		fmt.Println("len(b) != 4 or len(c) != 4 !!!")
	}

	g := make([]float64, 4)
	// h := make([]float64, 4)
	for i := 0; i < 4; i++ {
		g[i] = 1.0 / (math.Sqrt(32.0 * c[i]))
		// h[i] = b[i] / math.Sqrt(c[i])
	}

	item_1 := make([][][]float64, 4)
	item_2 := make([][][]float64, 4)
	item_3 := make([][][]float64, 4)
	item_4 := make([][][]float64, 4)
	for i := 0; i < 4; i++ {
		item_1[i] = MatrixChainMultiply_slice(coeffi_QKV.A_Q[i], Transp(coeffi_QKV.A_K[i]))
		item_1[i] = MultiplyByScalar(item_1[i], g[i])

		item_2[i] = MatrixChainMultiply_slice(coeffi_QKV.A_Q[i], Transp(coeffi_QKV.Constant_K[i]))
		item_2[i] = MultiplyByScalar(item_2[i], g[i])

		item_3[i] = MatrixChainMultiply_slice(coeffi_QKV.Constant_Q[i], Transp(coeffi_QKV.A_K[i]))
		item_3[i] = MultiplyByScalar(item_3[i], g[i])

		item_4[i] = MatrixChainMultiply_slice(coeffi_QKV.Constant_Q[i], Transp(coeffi_QKV.Constant_K[i]))
		item_4[i] = MultiplyByScalar(item_4[i], g[i])
		// item_4[i] = AddWithScalar(item_4[i], h[i])
	}

	return Coefficient_sqmax{
		// G_g: g,

		Item_1: item_1,
		Item_2: item_2,
		Item_3: item_3,
		Item_4: item_4,
	}
}

// func Compute_coefficient_head(coeffi_in Coefficient_input) (Coefficient_head) {

// 	W_Head_whole := MatrixChainMultiply_slice(coeffi_in.W_c, coeffi_in.Gamma_1, coeffi_in.W_1)

// 	return Coefficient_head{
// 		Sigma_1_diag: coeffi_in.Sigma_1_diag,

//			W_Head: SplitMatrixIntoFourChunks_byRow(W_Head_whole),
//		}
//	}
func GenerateCoefficient(dashModelParam utils.DashformerModelParameters) (Coefficient_dash, Coefficient_QKV, Coefficient_sqmax) {
	coeff_in := Create_Coefficient_input(dashModelParam)
	// coeff_in.PrintDimensions()

	coeff_dash := Compute_coefficient_dash(coeff_in)
	// coeff_dash.PrintDimensions()

	coeff_QKV := Compute_coefficient_QKV(coeff_in)
	// coeff_QKV.PrintDimensions()

	coeff_sqmax := Compute_coefficient_sqmax(coeff_QKV, dashModelParam.SoftMaxB, dashModelParam.SoftMaxC)
	// coeff_sqmax.PrintDimensions()

	return coeff_dash, coeff_QKV, coeff_sqmax
}

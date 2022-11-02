#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <torch/torch.h>
#include <functional>


#ifdef _DEBUG
#define EXEC oneapi::dpl::execution::seq
#else
#define EXEC oneapi::dpl::execution::par_unseq
#endif

#ifdef _DEBUG
#define PRINTT(x,str) std::cout<<#x<<x.sizes()<<x.dtype()<<" = "<<str<<std::endl;
#else
#define PRINTT(x,str)
#endif

namespace statistical
{
	std::vector<torch::Tensor> create_grads_out(const torch::Tensor &fl);
	/**
	 * @param vector<modules> the different modules configuration;
	 * @param vector<tensor> the different inputs
	 * @return  Tensor({D_PHI,D_PHI,N_PARAM}); the batched Fisher matrix of the model.	 * According to Abbas, A., Sutter, D., Zoufal, C., Lucchi, A., Figalli, A., and Woerner, S. (2021). The power of quantum neural networks. Nat Comput Sci, 1(6):403–409.
	 *
	 */
	template <typename Contained>
		torch::Tensor	get_Fisher_Matrix(std::vector<Contained> &models,
				std::vector<torch::Tensor> &xs)
		{
			const auto D_PHI= models.front()->parameters().size();
			const auto N_PARAM=models.size();
			const auto N_DATA=xs.front().size(0);


			std::vector<torch::Tensor> fls(N_PARAM);
			std::transform(EXEC,models.begin(),models.end(),xs.begin(),fls.begin(),[](Contained mdl, torch::Tensor x){
					mdl->train();
					auto fli= mdl(x);
					return fli;
					});
			const auto flT=torch::stack(fls,0);



			//PRINTT(flT,"Tensor(N_PARAM,N_DATA,SOUT)")
			/*fls(vector<Tensor(N_DATA,SOUT),N_PARAM>)*/


			std::vector<torch::Tensor> Log_fls(N_PARAM);
			std::transform(EXEC,fls.begin(),fls.end(),Log_fls.begin(),
					[](const auto & fl){
					return torch::log(fl);
					});

			const auto grd_outs=create_grads_out(Log_fls.front());
			/*grd_outs(Vector(Tensor(N_DATA,SOUT),N_DATA*SOUT))*/
			std::vector<torch::Tensor> dlpi(N_PARAM);
			std::transform(EXEC,Log_fls.begin(),Log_fls.end(),models.begin(),dlpi.begin(),
					[&grd_outs,&N_DATA](const auto & Lpi,auto & mdl){

					/*Lpi(Tensor(N_DATA,SOUT))*/
					/*parameters(Vector<Tensor(D_PHI/parameters.size())>)*/
					/*grd_outs(Vector<Tensor(N_DATA,SOUT),N_DATA*SOUT>)*/
					std::vector<torch::Tensor> gra(grd_outs.size());
					std::transform(EXEC,grd_outs.begin(),grd_outs.end(),gra.begin(),
							[&Lpi,&mdl](const auto & grad_out){
							auto var=torch::autograd::grad({Lpi},mdl->parameters(),{grad_out},true,false);
							std::for_each(EXEC,var.begin(),var.end(),[](auto &item){item=item.flatten();});
							auto glued=torch::cat(var);

							PRINTT(glued,"Tensor(D_PHI)")


							return glued;
							});
					/*gra(Vector<Tensor(D_PHI),N_DATA*SOUT>)*/
					auto linked=torch::stack(gra,0).reshape({N_DATA,(int64_t)gra.size()/N_DATA,gra.front().size(0)});

					//PRINTT(linked,"Tensor(N_DATA,SOUT,D_PHI)")



					return linked;
					});
			/*dlpi(Vector<Tensor(N_DATA,SOUT,D_PHI),N_PARAM>)*/
			auto dlpiT=torch::stack(dlpi,0);

			//PRINTT(dlpiT,"Tensor(N_PARAM,N_DATA,SOUT,D_PHI)")


			auto ddlpiT=torch::einsum("abcd,abce -> abcde",{dlpiT,dlpiT});
			auto Fisher_Matrix=torch::einsum("abcde,abc -> ade",{ddlpiT,flT})/N_DATA;

			//PRINTT(Fisher_Matrix,"Tensor(N_PARAM,D_PHI,D_PHI)")

			return Fisher_Matrix;
			/*Fisher_Matrix(Tensor(N_PARAM,D_PHI,D_PHI))*/
		}


	/**
	 * @param FM the Batched Fisher Matrix(Tensor(N_PARAM,D_PHI,D_PHI)).
	 * @return Tensor(N_PARAM,D_PHI), the normalized eigenvalues for each FM.
	 * One calculate the average Trace(aveTrace) of the eigenvalues over the set of FM.
	 * The normalized eigenvalues will result on the eigenvalues*D_PHI/aveTrace
	 *
	 *
	 */
	torch::Tensor   get_Normalized_Fisher_eig(torch::Tensor & FM );

	/**
	 * Calculate the effective dimension of a model using the normalized eigenvalues of the different Fisher matrices
	 * according to the reference above.
	 *@param NFME (Tensor(N_PARAM,D_PHI))the batched normalized eigenvalues of the Fisher Matrices.
	 *@param points (Tensor(Npoints)) . The number of data for which to calculate the Effective dimension.
	 *@param GAMMA (Tensor(Npoints)) . The parameter gamma on the reference above;
	 *@return Tensor(Npoints). The effective dimension for each point on points.
	 *
	 */
	torch::Tensor get_Effective_Dimension(torch::Tensor& NFME,torch::Tensor & points,const double& GAMMA);
};

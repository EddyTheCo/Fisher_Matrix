
#include"Fisher_Matrix.hpp"

namespace statistical
{

	std::vector<torch::Tensor> create_grads_out(const torch::Tensor &fl)
	{
		torch::NoGradGuard no_grad;

		std::vector<torch::Tensor> grads_out(fl.numel());
        std::vector<int64_t> index(fl.numel());
		std::iota(index.begin(),index.end(),0);
        std::transform(EXEC,index.begin(),index.end(),grads_out.begin(),
				[&fl](const auto & ind){
                auto indx=torch::tensor(ind);
				auto var=torch::one_hot(indx,fl.numel()).reshape(fl.sizes()).to(torch::kFloat64);
				return var;
				});
        return grads_out;

	}


	torch::Tensor   get_Normalized_Fisher_eig(torch::Tensor & FM )
	{
		torch::NoGradGuard no_grad;

		auto eig=torch::linalg::eigvals(FM);


    PRINTT(eig,"Tensor(N_PARAM,D_PHI)")
    assert(!(torch::imag(eig).norm().template item<double>()>0.0));

        eig=torch::real(eig);
		auto aveTrace=eig.sum()/eig.size(0);
        auto aveFM_eig=eig*eig.size(1)/aveTrace;

    PRINTT(aveFM_eig,"Tensor(N_PARAM,D_PHI)")

        return aveFM_eig;
	}

    torch::Tensor get_Effective_Dimension(torch::Tensor& NFME, at::Tensor &points, const double& GAMMA)
	{
		torch::NoGradGuard no_grad;

		auto gamma_n=points*GAMMA/(2*M_PI*torch::log(points));

    PRINTT(gamma_n,"Tensor(N_POINTS)")

        auto NewEig=torch::einsum("a,bc -> abc",{gamma_n,NFME});
		NewEig=1+NewEig;
		NewEig=torch::log(NewEig);

    PRINTT(NewEig,"Tensor(N_Points,N_PARAM,D_PHI)")


		NewEig=torch::sum(NewEig,2);

    PRINTT(NewEig,"Tensor(N_Points,N_PARAM)")


		NewEig=NewEig/2.0;

        auto Effe=2*(torch::logsumexp(NewEig,1)-log(NewEig.size(1)))/torch::log(gamma_n)/NFME.size(1);

    PRINTT(Effe,"Tensor(N_Points)")

		return Effe;

	}

}

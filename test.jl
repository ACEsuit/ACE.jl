using ACE, StaticArrays, ACE.SphericalHarmonics;
using ACE.SphericalHarmonics: index_y;
using ACE: evaluate
using LinearAlgebra

SH = SphericalHarmonics.SHBasis(5);

function PIbasis(ll::T, mm::T, R::SVector{N, Float64}) where{T,N}
    k = maximum(size(ll))
#    @show N
    A_part = 0;
    A = 1;
    for i = 1:k
        for j = 1:N/3
            Y = evaluate(SH, SVector(R.data[3*j-2:3*j]));
            A_part = A_part + Y[index_y(ll[i], mm[i])];
        end
        A = A * A_part;
        A_part = 0;
    end
    return A
end

function test_full_ribasis(ll::T, mm::T, R::SVector{N, Float64}) where{T,N}
    Cur_A = complex(0);
    a = Orb(0,0); b =Orb(0,0);
    μ_all = select_m(a,b,ll);
    no_basis = size(μ_all)[1]
    for t = 1 : no_basis
        mu = [i for i in μ_all[t].I];
        mu = reshape(mu,1,length(mu));
        @show mu
        Cur_A = Cur_A + CouCoe(ll,mm,mu) * PIbasis(ll, mu, R);
    end
    return Cur_A
end

function test_ss(C, ll::T, R::SVector{N, Float64}) where{T,N}
    Cur_A = complex(0);
    U = 0;
    a = Orb(0,0); b =Orb(0,0);
    p = a;
    q = b;
    Urc, μ_all = rc_basis(a,b,p,q,ll);
    #μ_all = select_m(p,q,ll);
    no_basis = size(μ_all)[1]
    for t = 1 : no_basis
        mu = [i for i in μ_all[t].I];
        mu = reshape(mu,1,length(mu));
        for j = 1 : size(Urc)[1]
            U = U + Urc[j,t];
#            Cur_A = Cur_A + Urc[j,t] * PIbasis(ll, mu, R);
        end
        Cur_A = Cur_A + U * PIbasis(ll, mu, R);
        U = 0;
    end
    Z = C*Cur_A;
end

function test_sp(C, ll::T, R::SVector{N, Float64}) where{T,N}
    Z = zeros(Complex{Float64},3);
    Cur_A = 0 + 0im
    U = 0;
    a = Orb(0,0); p = a;
        for i = 1:3
            q = Orb(1,i-2);
            for j = 1:3
                b = Orb(1,j-2);
                Urc, μ_all = rc_basis(a,b,p,q,ll);
                no_basis = size(μ_all)[1];
                for t = 1 : no_basis
                    mu = [i for i in μ_all[t].I];
                    mu = reshape(mu,1,length(mu));
                    # summation over i
                    for i_l = 1 : size(Urc)[1]
                        U = U + Urc[i_l,t];
                    end
                    Cur_A = Cur_A + U * PIbasis(ll, mu, R);
                    U = 0;
                end
                Z[i] = Z[i] + Cur_A
                Cur_A = 0;
            end
        end
        return Z'
end

function test_pp(C, ll::T, R::SVector{N, Float64}) where{T,N}
    Z = zeros(Complex{Float64},3,3);
    Cur_A = 0 + 0im
    U = 0;
#    Cur_A = zeros(Complex{Float64},3,3);
    for i1 = 1:3
        p = Orb(1,i1-2);
        for j1 = 1:3
            q = Orb(1,j1-2);
            for i = 1:3
                a = Orb(1,i-2);
                for j = 1:3
                    b = Orb(1,j-2);
#                    @show a,b,p,q
                    Urc, μ_all = rc_basis(a,b,p,q,ll);
#                    Urc = rc_basis(a,b,p,q,ll);
#                    μ_all = select_m(p,q,ll);
                    no_basis = size(μ_all)[1];
                    for t = 1 : no_basis
                        mu = [i for i in μ_all[t].I];
                        mu = reshape(mu,1,length(mu));
                        for i_l = 1 : size(Urc)[1]
                            U = U + Urc[i_l,t];
                        end
                        Cur_A = Cur_A + U * PIbasis(ll, mu, R);
                        U = 0;
                    end
                end
                Z[i1,i] = Z[i1,i] + Cur_A;
                Cur_A = 0;
            end
        end
    end
    return Z;
end

function Rot(R::SVector{N, Float64},Q) where {N}
    RotR = []; RotTemp = []; ii = 1;
#    K = randn(3, 3);
#    K = K - K';
#    Q = SMatrix{3,3}(rand([-1,1]) * exp(K)...);
    RotR = Q*R[3*ii-2:3*ii];
    if N/3 > 1
        for ii = 2:N/3
            RotTemp = SVector(R.data[3*ii-2:3*ii]);
            RotR = [RotR; Q*RotTemp];
        end
    end
    RotR = SVector(RotR)
    return RotR
end

function wigner_D(i::Int64, Q::N) where {N}
    if i == 1
        return 1;
    elseif i == 2;
        return [1 0 0; 0 1 0; 0 0 1];
    end
end










#   R = SVector(ones(N)...)
#   R = SVector(Float64(1), Float64(1), Float64(1));
#   K = randn(3, 3);
#   K = K - K';
#   Q = SMatrix{3,3}(rand([-1,1]) * exp(K)...);

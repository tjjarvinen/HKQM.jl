module Tullio_HKQM_ext

using Tullio
using HKQM

function Base.Array(pt::HKQM.PotentialTensor)
    x = Array(pt.x)
    y = Array(pt.y)
    z = Array(pt.z)
    @tullio out[i,j,k,I,J,K] := x[i,I,t] * y[j,J,t] * z[k,K,t] * pt.x.wt[t]
    return 2/sqrt(π) .* out
end


end
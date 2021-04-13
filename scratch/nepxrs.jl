using Base.Cartesian: @nexprs

@generated function runn(::Val{N}) where {N}
   quote
      @nexprs $N i -> ( @nexprs $N j -> println(i, "-", j) )
   end
end

runn(Val(10))

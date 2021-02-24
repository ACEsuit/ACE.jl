using Base.Cartesian: @nexprs

@generated function runn(::Val{N}) where {N}
   quote
      @nexprs $N i -> println(i)
   end
end

runn(Val(10))

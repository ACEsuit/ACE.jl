
struct Wrapper{NT}
   x::NT 
end 

@generated function syms(w::Wrapper{NamedTuple{SYMS, TT}}
                         ) where {SYMS, TT}
   code = "(" * prod( (":" * string(sym) * ",") for sym in SYMS ) * ")"
   quote
      $(Meta.parse(code))
   end
end

nt = (a = 1, b = 2, c = 5)
w = Wrapper(nt) 

syms(w)

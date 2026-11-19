module {
  hw.module @Counter(in %clock : !seq.clock, in %reset : i1, in %enable : i1, out observed : i8) {
    %zero = hw.constant 0 : i8
    %one = hw.constant 1 : i8
    %count = seq.firreg %next clock %clock reset sync %reset, %zero : i8
    %sum = comb.add %count, %one : i8
    %next = comb.mux %enable, %sum, %count : i8
    hw.output %count : i8
  }
}

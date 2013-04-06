#pragma rtGlobals=3		// Use modern global access method and strict wave access.

Menu "Macros"
	"Append Residuals...", AppendResidualsDialog()
End

Function AppendResiduals(ywave,xwave)
	String ywave, xwave
	if  (CmpStr("_calculated_",xwave) == 0)
		AppendToGraph/L=Lresid $ywave
	else 
		AppendToGraph/L=Lresid $ywave vs $xwave
	endif
	ModifyGraph nticks(Lresid)=2,standoff(bottom)=0, axisEnab(left)={0,0.7}
	ModifyGraph axisEnab(Lresid)={0.8,1}, freePos(Lresid)=0
	SetAxis/A/E=2 Lresid
	ModifyGraph mode(histResids)=2, lsize(histResids)=2
End

Function AppendResidualsDialog()
	String ywave, xwave
	Prompt ywave, "Residuals Data", popup WaveList("*",";","")
	Prompt xwave, "X Data", popup "_calculated_;"+WaveList("*",";","")
	DoPrompt "Append Residuals", ywave, xwave
	if (V_flag != 0)
		return -1;		//User canceled.
	endif
	AppendResiduals(ywave,xwave)
End
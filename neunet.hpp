NEUNET_BEGIN

class NetBase
{
protected:
    double dAcc = 1e-2, dNetLearnRate = 0;
    uint64_t iNetMiniBatch = 0, iNetDscType = GD_BGD;
    NET_LIST<LAYER_PTR> lsLayer;

    virtual void ValueAssign(NetBase &netSrc)
    {
        dAcc = netSrc.dAcc;
        dNetLearnRate = netSrc.dNetLearnRate;
        iNetMiniBatch = netSrc.iNetMiniBatch;
        iNetDscType = netSrc.iNetDscType;
    }
    void ShowIter() {}
    bool IterateFlag() { return true; }
    bool ForwProp() { return true; }
    bool BackProp() { return true; }
    set<vect> Deduce() { return blank_vect_seq; }
public:
    NetBase(NetBase &netSrc) : lsLayer(netSrc.lsLayer) { ValueAssign(netSrc); }
    NetBase(NetBase &&netSrc) : lsLayer(std::move(netSrc.lsLayer)) { ValueAssign(netSrc); }
    void operator=(NetBase &netSrc) { new (this)NetBase(netSrc); }
    void operator=(NetBase &&netSrc) { new (this)NetBase(std::move(netSrc)); }
    
    NetBase(uint64_t iDscType = GD_BGD, double dNetAcc = 1e-2, double dLearnRate = 0, uint64_t iMiniBatch = 0) : iNetDscType(iDscType), dAcc(dNetAcc), dNetLearnRate(dLearnRate), iNetMiniBatch(iMiniBatch) {}
    /* FC
    uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5
    * FC_BN
    double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dDmt = 1e-10
    * CONV
    * uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0
    * CONV_BN
    * uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dDmt = 1e-10
    * POOL
    * uint64_t iPoolTypeVal = POOL_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iActFuncTypeVal = NULL
    * LayerTrans - 
    * uint64_t iActFuncTypeVal = NULL
    * uint64_t iChannLnCnt, uint64_t iChannColCnt, uint64_t iActFuncIdx = NULL
     */
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) { return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...)); }
    uint64_t Depth() { return lsLayer.size(); }
    void Run() {}
};

class NetClassify : public NetBase
{
protected:
    bool bShowIterFlag = false;

    void ValueAssign(NetClassify &netSrc) { bShowIterFlag = netSrc.bShowIterFlag; }
    void IterShow(set<vect> &setPreOutput, set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        if(!setPreOutput.size()) setPreOutput = set<vect>(setOrigin.size());
        for(auto i=0; i<setOrigin.size(); i++)
        {
            if(!setPreOutput[i].is_matrix()) setPreOutput[i] = vect(setCurrOutput[i].LN_CNT, setCurrOutput[i].COL_CNT);
            for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j)
            {
                auto dCurrVal = setCurrOutput[i].pos_idx(j);
                if(setOrigin[i].pos_idx(j)) std::cout << '>';
                else std::cout << ' ';
                std::cout << dCurrVal << '\t';
                std::cout << j << '\t';
                double dif = dCurrVal - setPreOutput[i].pos_idx(j);
                setPreOutput[i].pos_idx(j) = dCurrVal;
                if(dif < 0) std::cout << dif;
                else std::cout << '+' << dif;
                std::cout << '\t';
                std::cout << setOrigin[i].pos_idx(j) << std::endl;
            }
            std::cout << std::endl;
        }
    }
    void IterShow(vect_t<vect> &batPreOutput, vect_t<vect> &batCurrOutput, vect_t<vect> &batOrigin) { for(auto i=0; i<batCurrOutput.size(); ++i) IterShow(batPreOutput[i], batCurrOutput[i], batOrigin[i]); }
    bool IterFlag(set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        for(auto i=0; i<setOrigin.size(); ++i) for(auto j=0; j<setOrigin[i].LN_CNT; ++j)
            if(std::abs(setOrigin[i][j][IDX_ZERO] - setCurrOutput[i][j][IDX_ZERO]) > dAcc) return true;
        return false;
    }
    bool IterFlag(vect_t<vect> &batCurrOutput, vect_t<vect> &batOrigin)
    {
        for(auto i=0; i<batCurrOutput.size(); ++i) if(IterFlag(batCurrOutput[i], batOrigin[i])) return true;
        return false;
    }
public:
    NetClassify(NetClassify &netSrc) : NetBase(netSrc) { ValueAssign(netSrc); }
    NetClassify(NetClassify &&netSrc) : NetBase(std::move(netSrc)) { ValueAssign(netSrc); }
    void operator=(NetClassify &netSrc) { new (this)NetClassify(netSrc); }
    void operator=(NetClassify &&netSrc) { new (this)NetClassify(std::move(netSrc)); }

    NetClassify(uint64_t iDscType = GD_BGD, double dNetAcc = 1e-2, double dLearnRate = 0, uint64_t iMiniBatch = 0, bool bShowIter = false) : NetBase(iDscType, dNetAcc, dLearnRate, iMiniBatch), bShowIterFlag(bShowIter) {}
};

NEUNET_END
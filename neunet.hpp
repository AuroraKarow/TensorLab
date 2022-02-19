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
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) { return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...)); }
    uint64_t Depth() { return lsLayer.size(); }
    void Run() {}
};

class NetClassify : public NetBase
{
protected:
    set<vect> setPreOutput;
    bool bShowIterFlag = false;

    void ValueAssign(NetClassify &netSrc) { bShowIterFlag = netSrc.bShowIterFlag; }
    void IterShow(set<vect> &setCurrOutput, set<vect> &setOrigin)
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
    void IterShow(vect_t<vect> &batCurrOutput, vect_t<vect> &batOrigin) { for(auto i=0; i<batCurrOutput.size(); ++i) IterShow(batCurrOutput[i], batOrigin[i]); }
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
    NetClassify(NetClassify &netSrc) : NetBase(netSrc), setPreOutput(netSrc.setPreOutput) { ValueAssign(netSrc); }
    NetClassify(NetClassify &&netSrc) : NetBase(std::move(netSrc)), setPreOutput(std::move(netSrc.setPreOutput)) { ValueAssign(netSrc); }
    void operator=(NetClassify &netSrc) { new (this)NetClassify(netSrc); }
    void operator=(NetClassify &&netSrc) { new (this)NetClassify(std::move(netSrc)); }

    NetClassify(uint64_t iDscType = GD_BGD, double dNetAcc = 1e-2, double dLearnRate = 0, uint64_t iMiniBatch = 0, bool bShowIter = false) : NetBase(iDscType, dNetAcc, dLearnRate, iMiniBatch), bShowIterFlag(bShowIter) {}
};

NEUNET_END
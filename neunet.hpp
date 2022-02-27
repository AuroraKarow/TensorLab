NEUNET_BEGIN

class NetBase
{
protected:
    double dAcc = 1e-2;
    uint64_t iNetDscType = GD_BGD;
    NET_LIST<LAYER_PTR> lsLayer;

    virtual void ValueAssign(NetBase &netSrc)
    {
        dAcc = netSrc.dAcc;
        iNetDscType = netSrc.iNetDscType;
    }
    void ShowIter() {}
    bool IterateFlag() { return true; }
    bool ForwProp() { return true; }
    bool BackProp() { return true; }
    set<vect> Deduce() { return blank_vect_seq; }
public:
    virtual void ValueCopy(NetBase &netSrc)
    {
        ValueAssign(netSrc);
        lsLayer = netSrc.lsLayer;
    }
    virtual void ValueMove(NetBase &&netSrc)
    {
        ValueAssign(netSrc);
        lsLayer = std::move(netSrc.lsLayer);
    }
    NetBase(NetBase &netSrc) { ValueCopy(netSrc); }
    NetBase(NetBase &&netSrc) { ValueMove(std::move(netSrc)); }
    void operator=(NetBase &netSrc) { ValueCopy(netSrc); }
    void operator=(NetBase &&netSrc) { ValueMove(std::move(netSrc)); }
    
    NetBase(uint64_t iDscType = GD_BGD, double dNetAcc = 1e-2) : iNetDscType(iDscType), dAcc(dNetAcc) {}
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
    virtual void ValueCopy(NetClassify &netSrc) { ValueAssign(netSrc); }
    virtual void ValueMove(NetClassify &&netSrc) { ValueAssign(netSrc); }
    NetClassify(NetClassify &netSrc) : NetBase(netSrc) { ValueCopy(netSrc); }
    NetClassify(NetClassify &&netSrc) : NetBase(std::move(netSrc)) { ValueMove(std::move(netSrc)); }
    void operator=(NetClassify &netSrc) { NetBase::operator=(netSrc); ValueCopy(netSrc); }
    void operator=(NetClassify &&netSrc) { NetBase::operator=(std::move(netSrc)); ValueMove(std::move(netSrc)); }

    NetClassify(uint64_t iDscType = GD_BGD, double dNetAcc = 1e-2, bool bShowIter = false) : NetBase(iDscType, dNetAcc), bShowIterFlag(bShowIter) {}
};

NEUNET_END
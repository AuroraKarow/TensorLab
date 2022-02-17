NEUNET_BEGIN

class NetBase
{
protected:
    double dAcc = 1e-2;
    uint64_t iMninBatchLastSize = 0;
    NET_LIST<LAYER_PTR> lsLayer;

    virtual void ValueAssign(NetBase &netSrc)
    {
        dAcc = netSrc.dAcc;
        iMninBatchLastSize = netSrc.iMninBatchLastSize;
    }
    void ShowIter() {}
    bool IterateFlag() { return true; }
    bool ForwProp() { return true; }
    bool BackProp() { return true; }
    set<vect> Deduce() { return blank_vect_seq; }
public:
    NetBase(double dNetAcc = 1e-2, uint64_t iMiniBatchSize = 0) : dAcc(dNetAcc) {}
    NetBase(NetBase &netSrc) : lsLayer(netSrc.lsLayer) { ValueAssign(netSrc); }
    NetBase(NetBase &&netSrc) : lsLayer(std::move(netSrc.lsLayer)) { ValueAssign(netSrc); }
    void operator=(NetBase &netSrc) { new (this)NetBase(netSrc); }
    void operator=(NetBase &&netSrc) { new (this)NetBase(std::move(netSrc)); }
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) { return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...)); }
    uint64_t Depth() { return lsLayer.size(); }
    void Run() {}
};

class NetClassify : public NetBase
{
protected:
    set<vect> setOutput, setOrigin;
    bool bShowIterFlag = false;

    void ValueAssign(NetClassify &netSrc) { bShowIterFlag = netSrc.bShowIterFlag; }
    void IterShow(set<vect> &setCurrOutput)
    {
        if(!setOutput.size()) setOutput = set<vect>(setOrigin.size());
        for(auto i=0; i<setOrigin.size(); i++)
        {
            if(!setOutput[i].is_matrix()) setOutput[i] = vect(setCurrOutput[i].LN_CNT, setCurrOutput[i].COL_CNT);
            for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j)
            {
                auto dCurrVal = setCurrOutput[i].pos_idx(j);
                if(setOrigin[i].pos_idx(j)) std::cout << '>';
                else std::cout << ' ';
                std::cout << dCurrVal << '\t';
                std::cout << j << '\t';
                double dif = dCurrVal - setOutput[i].pos_idx(j);
                setOutput[i].pos_idx(j) = dCurrVal;
                if(dif < 0) std::cout << dif;
                else std::cout << '+' << dif;
                std::cout << '\t';
                std::cout << setOrigin[i].pos_idx(j) << std::endl;
            }
            std::cout << std::endl;
        }
    }
    void IterShow(set<set<vect>> &batCurrOutput) { for(auto i=0; i<batCurrOutput.size(); ++i) IterShow(batCurrOutput[i]); }
    bool IterFlag(set<vect> &setCurrOutput)
    {
        for(auto i=0; i<setOrigin.size(); ++i) for(auto j=0; j<setOrigin[i].LN_CNT; ++j)
            if(std::abs(setOrigin[i][j][IDX_ZERO] - setCurrOutput[i][j][IDX_ZERO]) > dAcc) return true;
        return false;
    }
    bool IterFlag(set<set<vect>> &batCurrOutput)
    {
        for(auto i=0; i<batCurrOutput.size(); ++i) if(IterFlag(batCurrOutput[i])) return true;
        return false;
    }
public:
    NetClassify(NetClassify &netSrc) : NetBase(netSrc), setOutput(netSrc.setOutput), setOrigin(netSrc.setOrigin) { ValueAssign(netSrc); }
    NetClassify(NetClassify &&netSrc) : NetBase(std::move(netSrc)), setOutput(std::move(netSrc.setOutput)), setOrigin(std::move(netSrc.setOrigin)) { ValueAssign(netSrc); }
    void operator=(NetClassify &netSrc) { new (this)NetClassify(netSrc); }
    void operator=(NetClassify &&netSrc) { new (this)NetClassify(std::move(netSrc)); }

    NetClassify(double dNetAcc = 1e-2, bool bShowIter = false) : NetBase(dNetAcc), bShowIterFlag(bShowIter) {}
};

NEUNET_END
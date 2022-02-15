NEUNET_BEGIN

class NetBase
{
protected:
    double dAcc = 1e-2;
    NET_LIST<LAYER_PTR> lsLayer;

    void ShowIter() {}
    bool IterateFlag() { return true; }
    void ForwProp() {}
    void BackProp() {}
public:
    NetBase(double dNetAcc = 1e-2) : dAcc(dNetAcc) {}
    NetBase(NetBase &netSrc) : dAcc(netSrc.dAcc), lsLayer(netSrc.lsLayer) {}
    NetBase(NetBase &&netSrc) : dAcc(netSrc.dAcc), lsLayer(std::move(netSrc.lsLayer)) {}
    void operator=(NetBase &netSrc) { new (this)NetBase(netSrc); }
    void operator=(NetBase &&netSrc) { new (this)NetBase(std::move(netSrc)); }

    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) {return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...));}
    uint64_t Depth() { return lsLayer.size(); }
    void Run() {}
};

class NetClassify : public NetBase
{
protected:
    set<vect> setOutput, setOrigin;
    void ShowIter(set<vect> &setCurrOutput)
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
    bool IterateFlag(set<vect> &setCurrOutput)
    {
        for(auto i=0; i<setOrigin.size(); ++i) for(auto j=0; j<setOrigin[i].LN_CNT; ++j)
            if(std::abs(setOrigin[i][j][IDX_ZERO] - setCurrOutput[i][j][IDX_ZERO]) > dAcc) return true;
        return false;
    }
public:
    NetClassify(double dNetAcc = 1e-2) : NetBase(dNetAcc) {}
    NetClassify(NetClassify &netSrc) : NetBase(netSrc), setOutput(netSrc.setOutput), setOrigin(netSrc.setOrigin) {}
    NetClassify(NetClassify &&netSrc) : NetBase(std::move(netSrc)), setOutput(std::move(netSrc.setOutput)), setOrigin(std::move(netSrc.setOrigin)) {}
    void operator=(NetClassify &netSrc) { new (this)NetClassify(netSrc); }
    void operator=(NetClassify &&netSrc) { new (this)NetClassify(std::move(netSrc)); }
};

NEUNET_END
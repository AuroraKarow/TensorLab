NEUNET_BEGIN

class NetBase
{
protected:
    double dAcc = 1e-5;
    uint64_t iNetMiniBatch = 0;

    virtual void ValueAssign(NetBase &netSrc) { dAcc = netSrc.dAcc; iNetMiniBatch = netSrc.iNetMiniBatch; }
    void ShowIter() {}
    bool IterFlag() { return true; }
    bool ForwProp() { return true; }
    bool BackProp() { return true; }
    set<vect> Deduce() { return blank_vect_seq; }
public:
    virtual void ValueCopy(NetBase &netSrc) { ValueAssign(netSrc); }
    virtual void ValueMove(NetBase &&netSrc) { ValueAssign(netSrc); }
    NetBase(NetBase &netSrc) { ValueCopy(netSrc); }
    void operator=(NetBase &netSrc) { ValueCopy(netSrc); }
    
    NetBase(double dNetAcc = 1e-5, uint64_t iMinibatch = 0) : dAcc(dNetAcc), iNetMiniBatch(iMinibatch) {}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) { return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...)); }
    uint64_t Depth() { return 0; }
    bool Run() { return true; }
    void Reset() {}
    ~NetBase() { Reset(); }
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
    void IterShow(set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        for(auto i=0; i<setOrigin.size(); i++)
        {
            std::cout << " [BarY]\t\t[Label]\t[Y]" << std::endl;
            for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j)
            {
                if(setOrigin[i].pos_idx(j)) std::cout << '>';
                else std::cout << ' ';
                std::cout << setCurrOutput[i].pos_idx(j) << '\t';
                std::cout << j << '\t';
                std::cout << setOrigin[i].pos_idx(j) << std::endl;
            }
            std::cout << std::endl;
        }
    }
    uint64_t IterPass(set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        auto iPassCnt = 0;
        for(auto i=0; i<setOrigin.size(); i++) for(auto j=0; j<setOrigin[i].LN_CNT; ++j)
            if(setOrigin[i].pos_idx(j) && setCurrOutput[i].pos_idx(j) > (1-dAcc))
            {
                iPassCnt ++;
                break;
            }
        return iPassCnt;
    }
    bool IterFlag(set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        for(auto i=0; i<setCurrOutput.size(); ++i) for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j) if(setOrigin[i][j][IDX_ZERO]) if(std::abs(1-setCurrOutput[i][j][IDX_ZERO]) > dAcc) return true;
        return false;
    }
public:
    struct NetClassfyInput
    {
        set<feature> setInput;
        set<vect> setOrigin;
    };
    virtual void ValueCopy(NetClassify &netSrc) { ValueAssign(netSrc); }
    virtual void ValueMove(NetClassify &&netSrc) { ValueAssign(netSrc); }
    NetClassify(NetClassify &netSrc) : NetBase(netSrc) { ValueCopy(netSrc); }
    NetClassify(NetClassify &&netSrc) : NetBase(std::move(netSrc)) { ValueMove(std::move(netSrc)); }
    void operator=(NetClassify &netSrc) { NetBase::operator=(netSrc); ValueCopy(netSrc); }
    void operator=(NetClassify &&netSrc) { NetBase::operator=(std::move(netSrc)); ValueMove(std::move(netSrc)); }
    
    NetClassify(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bShowIter = false) : NetBase(dNetAcc, iMinibatch), bShowIterFlag(bShowIter) {}
};

NEUNET_END
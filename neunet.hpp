NEUNET_BEGIN

class basnet
{
protected:
    layer_list lsLayer;
    uint64_t iCurrIdx = 0;
    layer_ptr pCurrPtr;
public:
    uint64_t Depth() {return lsLayer.size();}
};

NEUNET_END
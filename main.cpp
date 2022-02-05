#include "dataset"
#include "neunet"

using namespace std;
using namespace dataset;
using namespace neunet;

class TestNet : public Basnet
{
protected:
    bool ForwProp(set<vect> &setInputVec, set<feature> &setInputFt, bool bDeduce = false)
    {
        auto head = lsLayer.head_node();
        setOutputVec = setInputVec;
        setOutputFt = setInputFt;
        while(head)
        {
            switch (head->data->iLayerType)
            {
            case FC:
                setOutputVec = LAYER_INSTANCE<LAYER_FC>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case FC_ADA:
                setOutputVec = LAYER_INSTANCE<LAYER_FC_ADA>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case CONV:
                setOutputFt = LAYER_INSTANCE<LAYER_CONV>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case CONV_ADA:
                setOutputFt = LAYER_INSTANCE<LAYER_CONV_ADA>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case POOL:
                setOutputFt = LAYER_INSTANCE<LAYER_POOL>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case BN_FC:
                if(bDeduce) setOutputVec = LAYER_INSTANCE<LAYER_BN_FC>(head->data) -> Deduce(iNetMiniBatch);
                else setOutputVec = LAYER_INSTANCE<LAYER_BN_FC>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case BN_FC_ADA:
                if(bDeduce) setOutputVec = LAYER_INSTANCE<LAYER_BN_FC_ADA>(head->data) -> Deduce(iNetMiniBatch);
                else setOutputVec = LAYER_INSTANCE<LAYER_BN_FC_ADA>(head->data) -> ForwProp(setOutputVec);
                if(setOutputVec.size()) break;
                else return false;
            case BN_CONV:
                if(bDeduce) setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV>(head->data) -> Deduce(iNetMiniBatch);
                else setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case BN_CONV_ADA:
                if(bDeduce) setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV_ADA>(head->data) -> Deduce(iNetMiniBatch);
                setOutputFt = LAYER_INSTANCE<LAYER_BN_CONV_ADA>(head->data) -> ForwProp(setOutputFt);
                if(setOutputFt.size()) break;
                else return false;
            case TRANS_TO_VECT:
                setOutputVec = LAYER_INSTANCE<LAYER_TRAN_TO_VECT>(head->data) -> ForwProp(setOutputFt);
                if(setOutputVec.size()) break;
                else return false;
            case TRANS_TO_FEAT:
                setOutputFt = LAYER_INSTANCE<LAYER_TRAN_TO_FEAT>(head->data) -> ForwProp(setOutputVec);
                if(setOutputFt.size()) break;
                else return false;
            default: return false;
            }
            head = head -> next();
        }
        return true;
    }
    bool BackProp(set<vect> &setOrigin)
    {
        auto tail = lsLayer.tail_node();
        if(bLastIsVect)  setGradVec = softmax_cec_grad(setOutputVec, setOrigin);
        else setGradVec = cec_grad(setOutputVec, setOrigin);
        while(tail)
        {
            switch (tail->data->iLayerType)
            {
            case FC:
                setGradVec = LAYER_INSTANCE<LAYER_FC>(tail->data) -> BackProp(setGradVec, dNetLearnRate);
                if(setGradVec.size()) break;
                else return false;
            case FC_ADA:
                setGradVec = LAYER_INSTANCE<LAYER_FC_ADA>(tail->data) -> BackProp(setGradVec);
                if(setGradVec.size()) break;
                else return false;
            case CONV:
                setGradFt = LAYER_INSTANCE<LAYER_CONV>(tail->data) -> BackProp(setGradFt, dNetLearnRate);
                if(setGradFt.size()) break;
                else return false;
            case CONV_ADA:
                setGradFt = LAYER_INSTANCE<LAYER_CONV_ADA>(tail->data) -> BackProp(setGradFt);
                if(setGradFt.size()) break;
                else return false;
            case POOL:
                setGradFt = LAYER_INSTANCE<LAYER_POOL>(tail->data) -> BackProp(setGradFt);
                if(setGradFt.size()) break;
                else return false;
            case BN_FC:
                setGradVec = LAYER_INSTANCE<LAYER_BN_FC>(tail->data) -> BackProp(setGradVec, dNetLearnRate);
                if(setGradVec.size()) break;
                else return false;
            case BN_FC_ADA:
                setGradVec = LAYER_INSTANCE<LAYER_BN_FC_ADA>(tail->data) -> BackProp(setGradVec);
                if(setGradVec.size()) break;
                else return false;
            case BN_CONV:
                setGradFt = LAYER_INSTANCE<LAYER_BN_CONV>(tail->data) -> BackProp(setGradFt, dNetLearnRate);
                if(setGradFt.size()) break;
                else return false;
            case BN_CONV_ADA:
                setGradFt = LAYER_INSTANCE<LAYER_BN_CONV_ADA>(tail->data) -> BackProp(setGradFt);
                if(setGradFt.size()) break;
                else return false;
            case TRANS_TO_VECT:
                setGradFt = LAYER_INSTANCE<LAYER_TRAN_TO_VECT>(tail->data) -> BackProp(setGradVec);
                if(setGradFt.size()) break;
                else return false;
            case TRANS_TO_FEAT:
                setGradVec = LAYER_INSTANCE<LAYER_TRAN_TO_FEAT>(tail->data) -> BackProp(setGradFt);
                if(setGradVec.size()) break;
                else return false;
            default: return false;
            }
            tail = tail -> prev();
        }
        return true;
    }
public:
    TestNet(bool bShowItern = false, double dAcc = 1e-5, double dLearnRate = 1e-10, uint64_t iMiniBatch = 0) : Basnet(bShowItern, dAcc, dLearnRate, iMiniBatch) {}
    bool Run(set<vect> &setInputVec, set<feature> &setInputFt, set<vect> &setOrigin)
    {
        do
        {
            if(!ForwProp(setInputVec, setInputFt) || !BackProp(setOrigin)) return false;
            if(bShowIteration) ShowIter(setOrigin);
        }
        while (IterateFlag(setInputVec, setInputFt, setOrigin));
        return true;
    }
};

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", 20, true, true, 2);
    TestNet net(true);
    /* Need to appoint the first layer
     * It needn't appoint current layer's activate function if next one is BN layer
     */
    net.AddLayer<LAYER_CONV_ADA>(6, 1, 5, 5, 1, 1, NULL_FUNC, CONV_ADA, true);
    // BN layer, appoint the previous layer's activate function.
    net.AddLayer<LAYER_BN_CONV_ADA>(6, RELU);
    net.AddLayer<LAYER_POOL>(2, 2, 2, 2);
    net.AddLayer<LAYER_CONV_ADA>(16, 6, 5, 5, 1, 1, NULL_FUNC);
    net.AddLayer<LAYER_BN_CONV_ADA>(16, RELU);
    net.AddLayer<LAYER_POOL>(2, 2, 2, 2);
    net.AddLayer<LAYER_CONV_ADA>(120, 16, 5, 5, 1, 1, NULL_FUNC);
    net.AddLayer<LAYER_BN_CONV_ADA>(120, RELU);
    net.AddLayer<LAYER_TRAN_TO_VECT>();
    net.AddLayer<LAYER_FC_ADA>(120, 84, NULL_FUNC);
    net.AddLayer<LAYER_BN_FC_ADA>(SIGMOID);
    net.AddLayer<LAYER_FC>(84, 10, SOFTMAX);
    net.Run(set<vect>(), dataset.elem, dataset.orgn());
    return EXIT_SUCCESS;
}
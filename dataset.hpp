DATASET_BEGIN

class MNIST
{
public:
    MNIST_SEQ_ELEM elem;
    MNIST_SEQ_LBL elem_lbl;
private:
    std::ifstream dat_stream;
    std::ifstream lbl_stream;
    uint64_t QNTY_STAT = 0;
    uint64_t LN_CNT = 0;
    uint64_t COL_CNT = 0;
    bool open_stream(std::string &dat_dir, std::string &lbl_dir)
    {
        dat_stream = std::ifstream(dat_dir, std::ios::in | std::ios::binary);
        lbl_stream = std::ifstream(lbl_dir, std::ios::in | std::ios::binary);
        return dat_stream.is_open() && lbl_stream.is_open();
    }
    // Close file stream, after procedure completing and must be called
    void close_stream()
    {
        dat_stream.close();
        lbl_stream.close();
    }
    // Magic number validation, after stream initialization
    bool magic_valid()
    {
        // validation magic number
        uint32_t magic = 0;
        // data
        dat_stream.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        auto dat_magic = bagrt::swap_endian(magic);
        if (dat_magic != MNIST_MAGIC_ELEM) return false;
        magic = 0;
        // label
        lbl_stream.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        auto lbl_magic = bagrt::swap_endian(magic);
        if (lbl_magic != MNIST_MAGIC_LBL) return false;
        return true;
    }
    // Item count, after the magic number validation
    uint64_t itm_cnt()
    {
        uint32_t dat_cnt = 0, lbl_cnt = 0;
        dat_stream.read(reinterpret_cast<char*>(&dat_cnt), sizeof(uint32_t));
        auto shrk_dat_cnt = bagrt::swap_endian(dat_cnt);
        lbl_stream.read(reinterpret_cast<char*>(&lbl_cnt), sizeof(uint32_t));
        auto shrk_lbl_cnt = bagrt::swap_endian(lbl_cnt);
        if(shrk_dat_cnt == shrk_lbl_cnt) return shrk_lbl_cnt;
        else return 0;
    }
    // Line and column count, after getting the item count, first call for getting line count and column the second. DO NOT CALL THIS FUNCTION MORE THAN 3 TIMES!
    uint64_t ln_col_size()
    {
        uint32_t _temp = 0;
        dat_stream.read(reinterpret_cast<char*>(&_temp), sizeof(uint32_t));
        return bagrt::swap_endian(_temp);
    }
    // Read data by current file stream pointer, calling once for getting a data unit with label synchronized.
    vect read_curr_dat(bool w_flag = true)
    {
        auto dat_size = LN_CNT * COL_CNT;
        char *dat_ptr = new char[dat_size];
        dat_stream.read(dat_ptr, dat_size);
        vect vec_dat;
        if(w_flag)
        {
            vec_dat = vect(LN_CNT, COL_CNT);
            for(auto i=0; i<LN_CNT; ++i)
                for(auto j=0; j<COL_CNT; ++j)
                    vec_dat[i][j] = (int)dat_ptr[i*LN_CNT + j];
        }
        delete []dat_ptr;
        dat_ptr = nullptr;
        return vec_dat;
    }
    // Read label by current file stream pointer, calling once for getting a label with data unit synchronized.
    uint64_t read_curr_lbl()
    {
        char label = 0;
        lbl_stream.read(&label, 1);
        return (uint64_t)label;
    }
    // Preprocess
    bool preprocess(std::string &dat_dir, std::string &lbl_dir)
    {
        if(open_stream(dat_dir, lbl_dir) && magic_valid())
        {
            QNTY_STAT = itm_cnt();
            if(QNTY_STAT)
            {
                LN_CNT = ln_col_size();
                COL_CNT = ln_col_size();
                return true;
            }
        }
        return false;
    }
    const bool is_bool;
public:
    MNIST(bool elem_bool) : is_bool(elem_bool) {}
    uint64_t size() {return elem.size();}
    uint64_t ln_cnt() {return LN_CNT;}
    uint64_t col_cnt() {return COL_CNT;}
    uint64_t elem_len() {return LN_CNT * COL_CNT;}
    bool valid() {return elem.size() == elem_lbl.size();}
    vect orgn(uint64_t lbl_val)
    {
        if(lbl_val < 10)
        {
            vect _orgn(MNIST_ORGN_SIZE, 1);
            _orgn[lbl_val][ZERO_IDX] = 1;
            return _orgn;
        }
        else return vect::blank_matrix();
    }
    set<vect> orgn()
    {
        set<vect> orgn_set(elem_lbl.size());
        for(auto i=0; i<elem_lbl.size(); ++i)
        {
            orgn_set[i] = vect(MNIST_ORGN_SIZE, SGL_IDX);
            orgn_set[i][elem_lbl[i]][ZERO_IDX] = 1;
        }
        return orgn_set;
    }
    bool load_data(std::string &dat_dir, std::string &lbl_dir, uint64_t load_qnty = 0, bool by_lbl = false)
    {
        bool pcdr_flag = true;
        if(preprocess(dat_dir, lbl_dir))
        {
            if(by_lbl && load_qnty)
            {
                bagrt::net_queue<uint64_t> qnty_list(MNIST_ORGN_SIZE);
                auto elem_cnt = 0;
                elem.init(load_qnty * MNIST_ORGN_SIZE);
                elem_lbl.init(load_qnty * MNIST_ORGN_SIZE);
                while(qnty_list.size())
                {
                    auto curr_lbl = read_curr_lbl();
                    if(qnty_list[curr_lbl] < load_qnty)
                    {
                        elem[elem_cnt].init();
                        elem[elem_cnt][ZERO_IDX] = read_curr_dat();
                        elem_lbl[elem_cnt ++] = curr_lbl;
                        ++ qnty_list[curr_lbl];
                    }
                    else qnty_list.erase(curr_lbl);
                }
            }
            else if(load_qnty)
            {
                bagrt::net_queue<uint64_t> lbl_data_stat(load_qnty);
                for(auto i=0; i<load_qnty; ++i) lbl_data_stat[i] = bagrt::random_number(0, QNTY_STAT);
                lbl_data_stat.sort();
                elem.init(load_qnty);
                elem_lbl.init(load_qnty);
                for(auto i=0,j=0; j<load_qnty; ++i)
                    if(i==lbl_data_stat[j])
                    {
                        
                        elem[j].init();
                        elem[j][ZERO_IDX] = read_curr_dat();
                        elem_lbl[j ++] = read_curr_lbl();
                    }
                    else
                    {
                        read_curr_dat(false);
                        read_curr_lbl();
                    }
            }
            else
            {
                load_qnty = QNTY_STAT;
                elem.init(load_qnty);
                elem_lbl.init(load_qnty);
                for(auto i=0; i<load_qnty; ++i)
                {
                    elem[i].init();
                    elem[i][ZERO_IDX] = read_curr_dat();
                    elem_lbl[i] = read_curr_lbl();
                }
            }
        }
        else pcdr_flag = false;
        close_stream();
        return pcdr_flag;
    }
    bool load_data(std::string &dat_dir, std::string &lbl_dir, bagrt::net_queue<uint64_t> qnty_list)
    {
        auto pcdr_flag = true;
        if(preprocess(dat_dir, lbl_dir))
        {
            auto elem_cnt = 0;
            while(qnty_list.size())
            {
                auto curr_lbl = read_curr_lbl();
                if(qnty_list[curr_lbl])
                {
                    elem[elem_cnt].init();
                    elem[elem_cnt][ZERO_IDX] = read_curr_dat();
                    elem_lbl[elem_cnt ++] = curr_lbl;
                    -- qnty_list[curr_lbl];
                }
                else qnty_list.erase(curr_lbl);
            }
        }
        else pcdr_flag = false;
        close_stream();
        return pcdr_flag;
    }
    MNIST(std::string dat_dir, std::string lbl_dir, bool elem_bool, uint64_t load_qnty = 0, bool by_lbl = false) : is_bool(elem_bool) {load_data(dat_dir, lbl_dir, load_qnty, by_lbl);}
    MNIST(std::string dat_dir, std::string lbl_dir, bool elem_bool, bagrt::net_queue<uint64_t> &qnty_list) : is_bool(elem_bool) {load_data(dat_dir, lbl_dir, qnty_list);}
    bool output_bitmap(std::string &dir_root, uint64_t extend, char div_syb = '\\')
    {
        if(is_bool || elem.size()!=BMIO_RGB_CNT) return false;
        else
        {
            auto cnt = 0;
            for(auto i=0; i<elem.size(); ++i)
            {
                auto name = '[' + std::to_string(elem_lbl[i]) + ']' + std::to_string(cnt++);
                bmio::bitmap img;
                img.set_raw(elem[i][BMIO_R], elem[i][BMIO_G], elem[i][BMIO_B]);
                if(!img.save_img(dir_root, name, extend, div_syb)) return false;
            }
            return true;
        }
    }
    // ~MNIST()
    // {
    //     QNTY_STAT = 0;
    //     LN_CNT = 0;
    //     COL_CNT = 0;
    // }
};

DATASET_END
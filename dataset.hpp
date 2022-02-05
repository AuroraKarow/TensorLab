DATASET_BEGIN

class MNIST
{
public:
    /* Data sequence */
    // Element list of data
    bagrt::net_queue<feature> elem;
    // Element index list of data, number sequence -> label value
    bagrt::net_queue<uint64_t> elem_lbl;
private:
    /* Magic number */
    // Data
    const uint64_t MAGIC_DATA_VALID = 2051;
    // Label
    const uint64_t MAGIC_LABEL_VALID = 2049;
    /* File stream */
    // Data
    std::ifstream dat_stream;
    // Label
    std::ifstream lbl_stream;
    /* Preprocess status */
    // Quantity status
    uint64_t QNTY_STAT = 0;
    // Line count
    uint64_t LN_CNT = 0;
    // Column count
    uint64_t COL_CNT = 0;
    /* Auxiliary function */
    // Open file stream for reading data and lable, must call before the whole procedure
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
        if (dat_magic != MAGIC_DATA_VALID) return false;
        magic = 0;
        // label
        lbl_stream.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        auto lbl_magic = bagrt::swap_endian(magic);
        if (lbl_magic != MAGIC_LABEL_VALID) return false;
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
    vect read_curr_dat(bool w_flag = true, uint64_t padding = 0, bool gray = false)
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
                {
                    int curr_pt = dat_ptr[i*LN_CNT + j];
                    if(curr_pt)
                        if(is_bool) vec_dat[i][j] = 1;
                        else if(gray) vec_dat[i][j] = 255;
                        else vec_dat[i][j] = curr_pt;
                    else vec_dat[i][j] = 0;
                }
        }
        delete []dat_ptr;
        dat_ptr = nullptr;
        if(padding) vec_dat = vec_dat.pad(padding, padding, padding, padding);
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
    // Origin vector dimension
    static const uint64_t ORGN_SIZE = 10;
    // Bool element flag
    const bool is_bool;
    // Check collection
    bool check = true;
public:
    /* Function */
    /**
     * @brief   Default constructor
     * @param   elem_bool   [Input] Data unit's element bool signal
     */
    MNIST(bool elem_bool) : is_bool(elem_bool) {}
    uint64_t size() {return elem.size();}
    // Column line per-data
    uint64_t ln_cnt() {return LN_CNT;}
    // Column count per-data
    uint64_t col_cnt() {return COL_CNT;}
    // Single data length
    uint64_t dat_len() {return ln_cnt() * col_cnt();}
    bool valid() {return elem.size() == elem_lbl.size();}
    /**
     * @brief   Get orignal vector of counterpart label
     * @param	lbl_val	[Input]	Label value
     * @return	Orignal vector
     */
    static vect orgn(uint64_t lbl_val)
    {
        if(lbl_val < 10)
        {
            vect _orgn(ORGN_SIZE, 1);
            _orgn[lbl_val][IDX_ZERO] = 1;
            return _orgn;
        }
        else return vect::blank_matrix();
    }
    // Get origin vector sequence
    bagrt::net_queue<vect> orgn()
    {
        if(size())
        {
            bagrt::net_queue<vect> elem_orgn(size());
            for(auto i=0; i<size(); ++i) elem_orgn[i] = orgn(elem_lbl[i]);
            return elem_orgn;
        }
        else return bagrt::net_queue<vect>::blank_queue();
    }
    /**
     * @brief   Load data
     * @param	dat_dir	    [Quote]	Data set directory
     * @param	lbl_dir	    [Quote]	Label set directory
     * @param	load_qnty	[Input]	Loading count
     * @param	by_lbl	    [Input]	Quantity for each label
     * @return	Data load validation
     * @retval  true    Load successfully
     * @retval  false   Load filed
     */
    bool load_data(std::string &dat_dir, std::string &lbl_dir, uint64_t load_qnty = 0, bool by_lbl = false, uint64_t padding = 0)
    {
        bool pcdr_flag = true;
        if(preprocess(dat_dir, lbl_dir))
        {
            if(by_lbl && load_qnty)
            {
                bagrt::net_queue<uint64_t> qnty_list(ORGN_SIZE);
                auto elem_cnt = 0;
                elem.init(load_qnty * ORGN_SIZE);
                elem_lbl.init(load_qnty * ORGN_SIZE);
                while(check)
                {
                    auto curr_lbl = read_curr_lbl();
                    if(qnty_list[curr_lbl] < load_qnty)
                    {
                        elem[elem_cnt].init();
                        elem[elem_cnt][IDX_ZERO] = read_curr_dat(true, padding);
                        elem_lbl[elem_cnt ++] = curr_lbl;
                        ++ qnty_list[curr_lbl];
                    }
                    else read_curr_dat(false);
                    auto check_cnt = 0;
                    for(check_cnt=0; check_cnt<ORGN_SIZE; ++check_cnt) if(qnty_list[check_cnt] < load_qnty) break;
                    check = check_cnt != ORGN_SIZE;
                }
            }
            else if(load_qnty)
            {
                bagrt::net_queue<uint64_t> lbl_data_stat(load_qnty);
                for(auto i=0; i<load_qnty; ++i) lbl_data_stat[i] = bagrt::random_number(0, QNTY_STAT, true);
                lbl_data_stat.sort();
                elem.init(load_qnty);
                elem_lbl.init(load_qnty);
                for(auto i=0,j=0; j<load_qnty; ++i)
                    if(i==lbl_data_stat[j])
                    {
                        elem[j].init();
                        elem[j][IDX_ZERO] = read_curr_dat(true, padding);
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
                    elem[i][IDX_ZERO] = read_curr_dat(true, padding);
                    elem_lbl[i] = read_curr_lbl();
                }
            }
        }
        else pcdr_flag = false;
        close_stream();
        return pcdr_flag;
    }
    /**
     * @brief   Load data
     * @param	dat_dir	    [Quote]	Data set directory
     * @param	lbl_dir	    [Quote]	Label set directory
     * @param	qnty_list	[Input]	Data loading quantity note list for each label
     * @return	Data load validation
     * @retval  true    Load successfully
     * @retval  false   Load filed
     */
    bool load_data(std::string &dat_dir, std::string &lbl_dir, bagrt::net_queue<uint64_t> qnty_list, uint64_t padding = 0)
    {
        auto pcdr_flag = true;
        if(preprocess(dat_dir, lbl_dir) && qnty_list.size()==ORGN_SIZE)
        {
            auto elem_cnt = 0;
            while(check)
            {
                auto curr_lbl = read_curr_lbl();
                if(qnty_list[curr_lbl])
                {
                    elem[elem_cnt].init();
                    elem[elem_cnt][IDX_ZERO] = read_curr_dat();
                    elem_lbl[elem_cnt ++] = curr_lbl;
                    -- qnty_list[curr_lbl];
                }
                else read_curr_dat(false);
                auto check_cnt = 0;
                for(check_cnt=0; check_cnt<ORGN_SIZE; ++check_cnt) if(qnty_list[check_cnt] > 0) break;
                check = check_cnt != ORGN_SIZE;
            }
        }
        else pcdr_flag = false;
        close_stream();
        return pcdr_flag;
    }
    /**
     * @brief   Initialization constructor
     * @param	dat_dir	    [Quote]	Data set directory
     * @param	lbl_dir	    [Quote]	Label set directory
     * @param   elem_bool   [Input] Data unit's element bool signal
     * @param	load_qnty	[Input]	Loading count
     * @param	by_lbl	    [Input]	Quantity for each label
     */
    MNIST(std::string dat_dir, std::string lbl_dir, uint64_t load_qnty = 0, bool elem_bool = false, bool by_lbl = false, uint64_t padding = 0) : is_bool(elem_bool) {load_data(dat_dir, lbl_dir, load_qnty, by_lbl, padding);}
    /**
     * @brief   Initialization constructor
     * @param	dat_dir	    [Quote]	Data set directory
     * @param	lbl_dir	    [Quote]	Label set directory
     * @param   elem_bool   [Input] Data unit's element bool signal
     * @param	qnty_list	[Input]	Data loading quantity note list for each label
     */
    MNIST(std::string dat_dir, std::string lbl_dir, bool elem_bool, bagrt::net_queue<uint64_t> &qnty_list, uint64_t padding = 0) : is_bool(elem_bool){load_data(dat_dir, lbl_dir, qnty_list, padding);}
    /**
     * @brief   Save data as bitmap
     * @param	dir_root	[Quote]	Saving root directory, '\\' is used to seperate sub directory path
     * @param   format      [Input] Image format, using bmio
     * [BMP][PNG][JPG][GIF][TIF]
     * @return	Save validation
     * @retval  true    Save successfully
     * @retval  false   Save filed
     */
    bool output_bitmap(std::string dir_root, uint64_t format = BMIO_BMP)
    {
        if(is_bool) return false;
        else
        {
            auto cnt = 0;
            for(auto i=0; i<elem.size(); ++i)
            {
                auto name = '[' + std::to_string(cnt++) + ']' + std::to_string(elem_lbl[i]);
                bmio::bitmap img;
                img.set_raw(elem[i][IDX_ZERO], elem[i][IDX_ZERO], elem[i][IDX_ZERO], elem[i][IDX_ZERO]);
                if(!img.save_img(dir_root, name, format)) return false;
            }
            return true;
        }
    }
    // ~MNIST() {}
};

DATASET_END
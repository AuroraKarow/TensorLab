ASYNC_BEGIN

class shared_signal
{
public:
    shared_signal(bool init_val = false) : sgn(init_val) {}
    shared_signal(shared_signal &src) : sgn(src.get_sgn()) {}
    void operator=(bool val) { set_sgn(val); }
    void operator=(shared_signal &src) { set_sgn(src.get_sgn()); }
    operator bool() { return get_sgn(); }
    bool get_sgn()
    {
        std::shared_lock<std::shared_mutex> lock(shrd_mtx);
        return sgn;
    }
    void set_sgn(bool tar_sgn)
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        sgn = tar_sgn;
    }
    void set_true() { set_sgn(true); }
    void set_false() { set_sgn(false); }
protected:
    mutable std::shared_mutex shrd_mtx;
    bool sgn = false;
};

class shared_counter
{
public:
    shared_counter(uint64_t init_val = 0) : cnt(init_val) {}
    shared_counter(shared_counter &src) : cnt(src.get_cnt()) {}
    void operator=(uint64_t val) { set_cnt(val); }
    void operator=(shared_counter &src) { set_cnt(src.get_cnt()); }
    shared_counter &operator++() { increment(); return *this; }
    shared_counter operator++(int) { auto temp = *this; increment(); return temp; }
    shared_counter &operator--() { decrement(); return *this; }
    shared_counter operator--(int) { auto temp = *this; decrement(); return temp; }
    operator uint64_t() { return get_cnt(); }
    uint64_t get_cnt()
    {
        std::shared_lock<std::shared_mutex> lock(shrd_mtx);
        return cnt;
    }
    void set_cnt(uint64_t tar_cnt = 0)
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        cnt = tar_cnt;
    }
    void increment()
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        ++ cnt;
    }
    void decrement()
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        -- cnt;
    }
protected:
    mutable std::shared_mutex shrd_mtx;
    uint64_t cnt = 0;
};

template<typename _rtnT, typename ... _parasT> std::function<_rtnT(_parasT...)> capsulate_function(_rtnT(*func_val)(_parasT...))
{
    std::function<_rtnT(_parasT...)> func_temp = static_cast<_rtnT(*)(_parasT...)>(func_val);
    return func_temp;
}
template<typename _rtnT, typename ... _parasT> auto package_function(_rtnT(*func_val)(_parasT...), _parasT ...paras) -> std::shared_ptr<std::packaged_task<_rtnT()>>
{
    return std::make_shared<std::packaged_task<_rtnT()>>(std::bind(std::forward<_rtnT(*)(_parasT...)>(func_val), std::forward<_parasT>(paras)...));
}

class async_batch
{
private:
    bagrt::net_queue<std::thread> td_set;
    bagrt::net_queue<std::function<void()>> tsk_set;
    shared_counter tsk_cnt;
    shared_signal stop;
    bagrt::net_queue<shared_signal> proc_set;
    std::mutex td_mtx_tsk, td_mtx_proc;
    std::condition_variable cond_tsk, cond_proc;
    uint64_t asyn_batch_size = ASYNC_CORE_CNT;
public:
    async_batch(uint64_t batch_size = ASYNC_CORE_CNT) : asyn_batch_size(batch_size), td_set(batch_size), tsk_set(batch_size), proc_set(batch_size)
    {
        for(auto i=0; i<asyn_batch_size; ++i) td_set[i] = std::thread([this](int idx){ while(true)
        {
            {
                std::unique_lock<std::mutex> lkTsk(td_mtx_tsk);
                while(!(proc_set[idx] || stop)) cond_tsk.wait(lkTsk);
            }
            if(stop) return;
            tsk_set[idx]();
            -- tsk_cnt;
            {
                std::unique_lock<std::mutex> lkProc(td_mtx_proc);
                proc_set[idx] = false;
                cond_proc.notify_all();
            }
        }}, i);
    }
    template<typename _func, typename ... _para> auto set_task(uint64_t bat_idx, _func &&func_val, _para &&...args) -> std::future<typename std::result_of<_func(_para...)>::type>
    {
        using ret_type = typename std::result_of<_func(_para...)>::type;
        auto p_curr_task = std::make_shared<std::packaged_task<ret_type()>>(std::bind(std::forward<_func>(func_val), std::forward<_para>(args)...));
        std::future<ret_type> res = p_curr_task->get_future();
        if(bat_idx < asyn_batch_size)
        {
            std::unique_lock<std::mutex> lkProc(td_mtx_proc);
            while(proc_set[bat_idx] && !stop) cond_proc.wait(lkProc);
            tsk_set[bat_idx] = [p_curr_task]() { (*p_curr_task)(); };
        }
        else stop = true;
        ++ tsk_cnt;
        {
            std::unique_lock<std::mutex> lkTsk(td_mtx_tsk);
            proc_set[bat_idx] = true;
            cond_tsk.notify_all();
        }
        return res;
    }
    uint64_t task_cnt() { return tsk_cnt; }
    uint64_t batch_size() { return asyn_batch_size; }
    ~async_batch()
    {
        stop = true;
        cond_tsk.notify_all();
        for(auto i=0; i<td_set.size(); ++i) if(td_set[i].joinable()) td_set[i].join();
    }
};

template<typename T> class async_queue
{
private:
    bagrt::net_link<T> ls_val;
    mutable std::shared_mutex tdmtx;
public:
    async_queue(async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = src.ls_val;
    }
    async_queue(async_queue &&src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = std::move(src.ls_val);
        src.reset();
    }
    void operator=(async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = src.ls_val;
    }
    void operator=(async_queue &&src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = std::move(src.ls_val);
        src.reset();
    }
    bool operator==(async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        return (this->ls_val == src.ls_val);
    }
    bool operator!=(async_queue &src) { return !(*this != src); }
    friend std::ostream &operator<<(std::ostream &output, async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        output << src.ls_val << std::endl;
        return output;
    }

    async_queue() {}
    uint64_t size()
    {
        std::shared_lock<std::shared_mutex> lck(tdmtx);
        return ls_val.size();
    }
    template<typename...args> bool en_queue(args &&...paras)
    {
        std::unique_lock<std::shared_mutex> lck(tdmtx);
        return ls_val.emplace_back(std::forward<args>(paras)...);
    }
    T de_queue()
    {
        std::unique_lock<std::shared_mutex> lck(tdmtx);
        return ls_val.erase(IDX_ZERO);
    }
    void reset()
    {
        std::unique_lock<std::shared_mutex> lck(tdmtx);
        ls_val.reset();
    }
    ~async_queue() { reset(); }
};

class async_tool
{
private:
    bagrt::net_queue<std::thread> td_set;
    async_queue<std::function<void()>> tsk_set;
    std::mutex td_mtx;
    std::condition_variable cond;
    shared_signal stop;
public:
    async_tool(uint64_t thread_size = ASYNC_CORE_CNT) : stop(false), td_set(thread_size) {
    for(auto i=0; i<td_set.size(); ++i) td_set[i] = std::thread([this]
    {
        while(true)
        {
            std::function<void()> curr_tsk;
            {
                std::unique_lock<std::mutex> lck(td_mtx);
                while(!(this->tsk_set.size() || stop)) cond.wait(lck);
                if(this->stop && !this->tsk_set.size()) return;
                curr_tsk = std::move(this->tsk_set.de_queue());
            }
            curr_tsk();
        }
    });}
    template<typename _funcT, typename ... _parasT> auto add_task(_funcT &&func_val, _parasT &&...paras) -> std::future<typename std::result_of<_funcT(_parasT ...)>::type>
    {
        // Thread function result type name (For deduce)
        using _rtnT = std::result_of<_funcT(_parasT...)>::type;
        // Function task package
        auto task = std::make_shared<std::packaged_task<_rtnT()>> (std::bind(std::forward<_funcT>(func_val), std::forward<_parasT>(paras)...));
        // Get task result
        std::future<_rtnT> rtn = task->get_future();
        {
            // Mutex lock
            std::unique_lock<std::mutex> lock(td_mtx);
            if(stop) throw std::runtime_error("Stop thread pool.");
            // Add task
            tsk_set.en_queue([task](){ (*task)(); });
        }
        cond.notify_one();
        return rtn;
    }
    ~async_tool()
    {
        stop = true;
        cond.notify_all();
        for(auto i=0; i<td_set.size(); ++i) if(td_set[i].joinable()) td_set[i].join();
    }
};

ASYNC_END
THRD_POOL_BEGIN

uint64_t thread_cnt() {return std::thread::hardware_concurrency();}

uint64_t thread_margin(uint64_t bat_size, uint64_t thrd_cnt = 0)
{
    auto i_thread_cnt = 0;
    if(!thrd_cnt) thrd_cnt = thread_cnt();
    return bat_size % thrd_cnt;
}

uint64_t thread_unit(uint64_t curr_bat_idx, uint64_t bat_size, uint64_t thrd_cnt = 0)
{
    if(!thrd_cnt) thrd_cnt = thread_cnt();
    if(curr_bat_idx+thrd_cnt<bat_size) return thrd_cnt;
    else return thread_margin(bat_size, thrd_cnt);
}

class thread_pool
{
private:
    // Threads set
    std::vector<std::thread> thrds_set;
    // Tasks set
    std::queue<std::function<void()>> tasks_set;
    // Mutex
    std::mutex sync_mutex;
    // Condition
    std::condition_variable cond_val;
    // Processing signal
    bool stop;
public:
    inline thread_pool(uint64_t thread_cnt = thread_cnt()) : stop(false)
    {
        for(size_t i = 0;i<thread_cnt;++i)
        thrds_set.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->sync_mutex);
                        this->cond_val.wait(lock, [this]{ return this->stop || !this->tasks_set.empty(); });
                        if(this->stop && this->tasks_set.empty()) return;
                        task = std::move(this->tasks_set.front());
                        this->tasks_set.pop();
                    }
                    task();
                }
            }
        );
    }
    
    template<class _funcT, class ... _parasT> auto add_task(_funcT &&func_val, _parasT &&...paras)
        -> std::future<typename std::result_of<_funcT(_parasT ...)>::type>
    {
        using return_type = typename std::result_of<_funcT(_parasT...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<_funcT>(func_val), std::forward<_parasT>(paras)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(sync_mutex);
            // don't allow enqueueing after stopping the pool
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_set.emplace([task](){ (*task)(); });
        }
        cond_val.notify_one();
        return res;
    }
    // Destructor
    inline ~thread_pool()
    {
        // End whole procedure
        {
            std::unique_lock<std::mutex> lock(sync_mutex);
            stop = true;
        }
        cond_val.notify_all();
        // Join the thread
        for(std::thread &worker: thrds_set) worker.join();
    }
};

THRD_POOL_END
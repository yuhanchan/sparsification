#ifndef CMDARGS_H
#define CMDARGS_H

#include <string>
#include <vector>
#include <time.h>

/**
 * Our custom command line argument parsing class.
 */
class CmdArgs
{
public:
    CmdArgs(int argc, char **argv);
    const std::string &inFname() const { return _inFname; }
    const std::string &outFname() const { return _outFname; }
    const std::string &mode() const { return _mode; }
    bool success() const { return _success; }
    void dispHelp() const;
private:
    std::string _inFname; 
    std::string _outFname; 
    std::string _mode;
    bool _success;
};

#endif
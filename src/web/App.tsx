import { useState, useEffect, useRef, Component, createRef } from 'react'
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import ButtonGroup from '@mui/material/ButtonGroup';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import LinearProgress from '@mui/material/LinearProgress';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Accordion from '@mui/material/Accordion';
import FormControl from '@mui/material/FormControl';
import AccordionActions from '@mui/material/AccordionActions';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Paper from '@mui/material/Paper';
import HistoryEdu from '@mui/icons-material/HistoryEdu';
import GitHub from '@mui/icons-material/GitHub';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Code from '@mui/icons-material/Code';
import './App.css'
import Menu from '@mui/material/Menu';
import { LoremIpsum, ILoremIpsumOptions } from '../../src/ts/lorem-ipsum';


interface ILoremIpsumInBg {
    done: boolean;
    canceled: boolean;
    progress: number;
    cancel: () => void;
}


function loremIpsumInBg(options: ILoremIpsumOptions, length: number, update: (text: string[] | string) => void): ILoremIpsumInBg {
    let cancel = false;
    let obj: ILoremIpsumInBg = {
        done: false,
        canceled: false,
        progress: 0,
        cancel: () => {
            obj.canceled = true;
            cancel = true;
        }
    }
    async function inner(): Promise<void> {
        await new Promise(resolve => setTimeout(resolve, 1));
        let ipsum = new LoremIpsum(options);
        let remaining = length - 1;
        while (remaining >= 0 && !cancel) {
            let endTime = Date.now() + 50;
            let updateText: string[] | string;
            if (options.paragraphs) {
                updateText = [];
                while (remaining >= 0 && !cancel && Date.now() < endTime) {
                    let paragraph = '';
                    while (remaining >= 0) {
                        let letter = ipsum.next(remaining);
                        paragraph += letter;
                        remaining--;
                        if (letter === '\n') {
                            break;
                        }
                    }
                    updateText.push(paragraph);
                }
            } else {
                updateText = '';
                while (remaining >= 0) {
                    let letter = ipsum.next(remaining);
                    updateText += letter;
                    remaining--;
                    if (letter === '.' && (Date.now() >= endTime || cancel)) {
                        break;
                    }
                }
            }
            await new Promise(resolve => setTimeout(resolve, 2));
            obj.progress = (length - remaining) / length;
            obj.done = remaining < 0 || cancel;
            update(updateText);
        }
    }
    inner();
    return obj;
}

enum GeneratingState {
    NotStarted = 'NotStarted',
    Generating = 'Generating',
    Finished = 'Finished',
};

interface IState {
    generatingState: GeneratingState;
    progress: number;
};

const initialState: IState = {
    generatingState: GeneratingState.NotStarted,
    progress: 0,
};

type StateRef = React.RefObject<{
    state: IState;
    setState: React.Dispatch<React.SetStateAction<IState>>;
}>;

class App extends Component {

    loremIpsumState: ILoremIpsumInBg | undefined = undefined;
    state: IState;
    resultContent: Node[];
    resultContainerRef: React.RefObject<HTMLDivElement | null>;

    constructor(props: {}) {
        super(props);
        this.state = initialState;
        this.resultContent = [];
        this.resultContainerRef = createRef<HTMLDivElement>();
    }

    private refreshResultContainer() {
        if (this.resultContainerRef.current && this.resultContainerRef.current.childNodes.length === 0 && this.resultContent.length > 0) {
            for (let child of this.resultContent) {
                this.resultContainerRef.current.appendChild(child);
            }
        }
    }

    componentDidMount() {
        this.refreshResultContainer();
    }

    private appendNode(node: Node) {
        this.resultContent.push(node);
        if (this.resultContainerRef.current) {
            this.resultContainerRef.current.appendChild(node);
        }
    }

    private handleGenerate() {
        this.resultContent = [];
        this.resultContainerRef.current?.replaceChildren();
        this.loremIpsumState = loremIpsumInBg({ paragraphs: true, language: 'pl' }, 10000, (text: string[] | string) => {
            this.refreshResultContainer();
            if (typeof text === 'string') {
                this.appendNode(document.createTextNode(text));
            } else {
                for (let paragraph of text) {
                    let p = document.createElement('p');
                    p.appendChild(document.createTextNode(paragraph));
                    this.appendNode(p);
                }
            }
            this.setState(prev => ({ ...prev, progress: this.loremIpsumState?.progress || 1 }));
            if (this.loremIpsumState?.done) {
                this.setState(prev => ({ ...prev, generatingState: GeneratingState.Finished }));
                this.loremIpsumState = undefined;
            }
        });
        this.setState(prev => ({ ...prev, generatingState: GeneratingState.Generating, progress: 0 }));
    }

    private handleCancel() {
        this.loremIpsumState?.cancel();
    }

    render() {
        let state = this.state;
        return (<>
            <AppBar position="static">
                <Toolbar>
                    <IconButton
                        size="large"
                        edge="start"
                        color="inherit"
                        aria-label="menu"
                        sx={{ mr: 2 }}
                    >
                        <HistoryEdu />
                    </IconButton>
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Lorem ipsum
                        <Typography variant="subtitle2" component="div" sx={{ flexGrow: 4 }}>
                            Deterministic Lorem Ipsum generator powered by a minimal neural network
                        </Typography>
                    </Typography>
                    &nbsp;&nbsp;&nbsp;
                    <Button color="inherit" startIcon={<Code />}>Developers</Button>
                    &nbsp;&nbsp;&nbsp;
                    <Button color="inherit" startIcon={<GitHub />}>GutHub</Button>
                </Toolbar>
            </AppBar>

            <Accordion style={{ maxWidth: '840px', margin: '30px auto' }} elevation={3}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="panel1-content"
                    id="panel1-header"
                >
                    <Typography variant='h6'>Options</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse
                    malesuada lacus ex, sit amet blandit leo lobortis eget.
                </AccordionDetails>
            </Accordion>

            <Paper elevation={3} style={{ maxWidth: '800px', margin: '30px auto', padding: '20px' }}>
                {(state.generatingState === GeneratingState.NotStarted) &&
                    <div style={{ textAlign: 'center' }}>
                        <Button variant='contained' color='secondary' size='large' style={{ width: 250 }} onClick={() => this.handleGenerate()}>Generate</Button>
                    </div>
                }
                {(state.generatingState === GeneratingState.Generating) &&
                    <>
                        <div style={{ textAlign: 'center' }}>
                            <Button variant='contained' color='error' size='large' style={{ width: 250 }} onClick={() => this.handleCancel()}>Cancel</Button>
                        </div>
                        <LinearProgress variant="determinate" value={state.progress * 100} style={{ marginTop: 20 }} />
                    </>
                }
                {(state.generatingState === GeneratingState.Finished) &&
                    <div style={{ textAlign: 'center' }}>
                        <Button variant='contained' color='secondary' size='large' style={{ width: 200 }} onClick={() => this.handleGenerate()}>Regenerate</Button>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                        <Button variant='contained' color='primary' size='large' style={{ width: 200 }} onClick={() => this.handleGenerate()}>Copy to clipboard</Button>
                    </div>
                }

            </Paper>

            <Paper elevation={3} style={{ maxWidth: '800px', margin: '30px auto 50px', padding: '20px', visibility: state.generatingState === GeneratingState.NotStarted ? 'hidden' : 'visible' }}>
                <div ref={this.resultContainerRef} style={{ textAlign: 'justify' }}></div>
            </Paper>

        </>)
    }
}

export default App

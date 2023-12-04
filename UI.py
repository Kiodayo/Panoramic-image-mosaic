import wx


# subclass of wx.Frame
class Frame(wx.Frame):
    def __init__(self, title):
        wx.Frame.__init__(
            self, None, title=title, pos=(
                150, 150), size=(
                1200, 1000))
        # Add Event Handler that will handle the wx.EVT_CLOSE
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        menuBar = wx.MenuBar()

        menu = wx.Menu()

        m_exit = menu.Append(
            wx.ID_EXIT,
            "E&xit\tAlt-X",
            "Close window and exit program.")
        # event | handler | source
        self.Bind(wx.EVT_MENU, self.OnClose, m_exit)

        menuBar.Append(menu, "&File")

        self.SetMenuBar(menuBar)

        self.pic1_button = wx.Button(
            self, wx.ID_ANY, u"第一张图片", wx.Point(
                300, 400), wx.DefaultSize, 0)
        self.pic1_button.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_INFOBK ) )

        panel = wx.Panel(self)

        panel.Layout()

    def OnClose(self, event):
        dlg = wx.MessageDialog(
            self,
            "Do you really want to close this application?",
            "Confirm Exit",
            wx.OK | wx.CANCEL | wx.ICON_QUESTION)
        # display the dialog and wait for clicking button
        # The result is either wx.ID_OK or wx.ID_CANCEL.
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.Destroy()


app = wx.App(redirect=True)
top = Frame("Hello World")
top.Show()
app.MainLoop()

import streamlit as st
from view_page import StockDashboard
from model_page import StockModelPage
from portfolio_page import PortfolioPage

def main():
    st.set_page_config(layout='wide', page_title='Stock Analysis', page_icon=':dollar:')
    page = st.sidebar.radio('Pages', ['View Page', 'Model Page', 'Portfolio Management'])
    if page == 'View Page':
       StockDashboard().run()
    elif page == 'Model Page':
       StockModelPage().run()
    elif page == 'Portfolio Management':
       PortfolioPage().run()

if __name__ == '__main__':
    main()
